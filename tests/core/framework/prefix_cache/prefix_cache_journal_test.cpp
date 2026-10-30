/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/prefix_cache/prefix_cache_journal.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace xllm {

class PrefixCacheJournalTestPeer {
 public:
  static void publish_stored(PrefixCacheJournal& journal, const XXH3Key& hash) {
    journal.publish_stored(hash, BlockType::KV);
  }

  static void publish_removed(PrefixCacheJournal& journal,
                              const XXH3Key& hash) {
    journal.publish_removed(hash, BlockType::KV);
  }

  static void publish_cleared(PrefixCacheJournal& journal) {
    journal.publish_cleared();
  }

  static PrefixCacheEvents events(PrefixCacheJournal& journal,
                                  size_t max_events) {
    return journal.get_prefixcache_events(max_events);
  }

  static PrefixCacheSnapshot snapshot(PrefixCacheJournal& journal) {
    return journal.get_snapshot();
  }

  static size_t pending(const PrefixCacheJournal& journal) {
    return journal.pending_events_.load(std::memory_order_relaxed);
  }

  static size_t resident(const PrefixCacheJournal& journal) {
    return journal.resident_.size();
  }
};

namespace {

XXH3Key make_hash(uint64_t value) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> bytes{};
  for (size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<uint8_t>((value >> ((index % 8) * 8)) ^ index);
  }
  return XXH3Key(bytes.data());
}

std::string hash_key(const XXH3Key& hash) {
  return std::string(reinterpret_cast<const char*>(hash.data),
                     sizeof(hash.data));
}

TEST(PrefixCacheJournalTest, DrainedEventsStayOutOfTheNextBatch) {
  PrefixCacheJournal journal;
  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(1));
  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(2));

  const PrefixCacheEvents first =
      PrefixCacheJournalTestPeer::events(journal, 16);
  ASSERT_EQ(first.events.size(), 2u);
  EXPECT_EQ(first.commit_version, 1u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 2u);

  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(3));
  const PrefixCacheEvents next =
      PrefixCacheJournalTestPeer::events(journal, 16);
  ASSERT_EQ(next.events.size(), 1u);
  EXPECT_EQ(next.commit_version, 2u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::pending(journal), 0u);
}

TEST(PrefixCacheJournalTest, SizesLargeBatchIncrementally) {
  PrefixCacheJournal journal;
  constexpr uint64_t kEvents = 256;
  for (uint64_t index = 0; index < kEvents; ++index) {
    const XXH3Key hash = make_hash(index + 1);
    PrefixCacheJournalTestPeer::publish_stored(journal, hash);
  }

  const PrefixCacheEvents batch =
      PrefixCacheJournalTestPeer::events(journal, kEvents);
  ASSERT_EQ(batch.events.size(), kEvents);
  EXPECT_EQ(batch.commit_version, 1u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), kEvents);
}

TEST(PrefixCacheJournalTest, PublishesIdleHighWatermarkKeepalive) {
  PrefixCacheJournal journal;
  const PrefixCacheEvents keepalive =
      PrefixCacheJournalTestPeer::events(journal, 8);
  EXPECT_EQ(keepalive.commit_version, 0u);
  EXPECT_TRUE(keepalive.events.empty());

  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(1));
  const PrefixCacheEvents events =
      PrefixCacheJournalTestPeer::events(journal, 8);
  static_cast<void>(events);

  const PrefixCacheEvents idle = PrefixCacheJournalTestPeer::events(journal, 8);
  EXPECT_TRUE(idle.events.empty());
  EXPECT_EQ(idle.commit_version, 1u);
}

TEST(PrefixCacheJournalTest, PendingCountDoesNotRejectLaterEvents) {
  PrefixCacheJournal journal;
  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(1));
  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(2));

  EXPECT_EQ(PrefixCacheJournalTestPeer::pending(journal), 2u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 0u);

  // The queue has no cap. Both events stay queued, and one drain applies them.
  const PrefixCacheEvents batch =
      PrefixCacheJournalTestPeer::events(journal, 16);
  EXPECT_EQ(batch.events.size(), 2u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::pending(journal), 0u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 2u);
}

TEST(PrefixCacheJournalTest, RemoveAndClearTrackTrueResidentState) {
  PrefixCacheJournal journal;
  const XXH3Key first = make_hash(1);
  const XXH3Key second = make_hash(2);
  PrefixCacheJournalTestPeer::publish_stored(journal, first);
  PrefixCacheJournalTestPeer::publish_stored(journal, second);
  PrefixCacheJournalTestPeer::publish_removed(journal, first);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 0u);

  const PrefixCacheEvents batch =
      PrefixCacheJournalTestPeer::events(journal, 16);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 1u);
}

TEST(PrefixCacheJournalTest, ExactRepeatedStoreDoesNotCloneResidentEntry) {
  PrefixCacheJournal journal;
  const XXH3Key original = make_hash(1);
  PrefixCacheJournalTestPeer::publish_stored(journal, original);
  PrefixCacheJournalTestPeer::publish_stored(journal, original);
  const PrefixCacheEvents batch =
      PrefixCacheJournalTestPeer::events(journal, 16);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 1u);

  const PrefixCacheSnapshot snapshot =
      PrefixCacheJournalTestPeer::snapshot(journal);

  ASSERT_EQ(snapshot.hashes.size(), 1u);
  EXPECT_EQ(hash_key(snapshot.hashes[0]), hash_key(original));
}

TEST(PrefixCacheJournalTest, SnapshotCopiesResidentVersionBeforeLaterEvents) {
  PrefixCacheJournal journal;
  for (uint64_t index = 0; index < 5; ++index) {
    const XXH3Key hash = make_hash(index + 1);
    PrefixCacheJournalTestPeer::publish_stored(journal, hash);
  }
  const PrefixCacheEvents committed =
      PrefixCacheJournalTestPeer::events(journal, 16);

  const PrefixCacheSnapshot snapshot =
      PrefixCacheJournalTestPeer::snapshot(journal);

  EXPECT_EQ(snapshot.hashes.size(), 5u);
  EXPECT_EQ(snapshot.commit_version, 1u);

  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(99));
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 5u);
  EXPECT_EQ(snapshot.hashes.size(), 5u);
}

TEST(PrefixCacheJournalTest,
     LargeSnapshotRetainsEntriesAcrossConcurrentMutation) {
  PrefixCacheJournal journal;
  constexpr uint64_t kEntries = 1024;
  std::set<std::string> expected_entries;
  for (uint64_t index = 0; index < kEntries; ++index) {
    const XXH3Key hash = make_hash(index + 1);
    expected_entries.insert(hash_key(hash));
    PrefixCacheJournalTestPeer::publish_stored(journal, hash);
  }
  const PrefixCacheEvents committed =
      PrefixCacheJournalTestPeer::events(journal, kEntries);

  std::thread writer([&journal]() {
    for (uint64_t index = 0; index < 256; ++index) {
      const XXH3Key removed = make_hash(index + 1);
      const XXH3Key stored = make_hash(10000 + index);
      PrefixCacheJournalTestPeer::publish_removed(journal, removed);
      PrefixCacheJournalTestPeer::publish_stored(journal, stored);
    }
  });
  const PrefixCacheSnapshot snapshot =
      PrefixCacheJournalTestPeer::snapshot(journal);
  writer.join();

  EXPECT_EQ(snapshot.commit_version, committed.commit_version);
  std::set<std::string> actual_entries;
  for (const XXH3Key& hash : snapshot.hashes) {
    actual_entries.insert(hash_key(hash));
  }
  EXPECT_EQ(actual_entries.size(), kEntries);
  EXPECT_EQ(actual_entries, expected_entries);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), kEntries);
}

TEST(PrefixCacheJournalTest, ClearIsVisibleOnlyAfterTheBatchCommits) {
  PrefixCacheJournal journal;
  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(1));
  const PrefixCacheEvents stored =
      PrefixCacheJournalTestPeer::events(journal, 8);
  const PrefixCacheSnapshot before =
      PrefixCacheJournalTestPeer::snapshot(journal);
  EXPECT_EQ(before.hashes.size(), 1u);

  PrefixCacheJournalTestPeer::publish_cleared(journal);
  const PrefixCacheSnapshot queued =
      PrefixCacheJournalTestPeer::snapshot(journal);
  EXPECT_EQ(queued.hashes.size(), 1u);

  const PrefixCacheEvents cleared =
      PrefixCacheJournalTestPeer::events(journal, 8);
  const PrefixCacheSnapshot after =
      PrefixCacheJournalTestPeer::snapshot(journal);
  EXPECT_TRUE(after.hashes.empty());
  EXPECT_EQ(after.commit_version, stored.commit_version + 1);
}

TEST(PrefixCacheJournalTest, ConcurrentClearLeavesCommittedSnapshotIntact) {
  PrefixCacheJournal journal;
  std::set<std::string> expected;
  for (uint64_t index = 0; index < 128; ++index) {
    const XXH3Key hash = make_hash(index + 1);
    expected.insert(hash_key(hash));
    PrefixCacheJournalTestPeer::publish_stored(journal, hash);
  }
  const PrefixCacheEvents committed =
      PrefixCacheJournalTestPeer::events(journal, 256);

  std::atomic<bool> start = false;
  PrefixCacheSnapshot snapshot;
  std::thread snapshot_thread([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    snapshot = PrefixCacheJournalTestPeer::snapshot(journal);
  });
  std::thread clear_thread([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    PrefixCacheJournalTestPeer::publish_cleared(journal);
  });
  start.store(true, std::memory_order_release);
  snapshot_thread.join();
  clear_thread.join();

  EXPECT_EQ(snapshot.commit_version, committed.commit_version);
  EXPECT_EQ(snapshot.hashes.size(), 128u);
  for (const XXH3Key& hash : snapshot.hashes) {
    EXPECT_NE(expected.find(hash_key(hash)), expected.end());
  }

  const PrefixCacheEvents cleared_batch =
      PrefixCacheJournalTestPeer::events(journal, 8);
  const PrefixCacheSnapshot after_clear =
      PrefixCacheJournalTestPeer::snapshot(journal);
  EXPECT_TRUE(after_clear.hashes.empty());
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 0u);
}

TEST(PrefixCacheJournalTest, ResidentSetKeepsOneCopyOfARepeatedHash) {
  PrefixCacheJournal journal;
  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(1));
  PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(1));
  const PrefixCacheEvents batch =
      PrefixCacheJournalTestPeer::events(journal, 16);
  EXPECT_EQ(PrefixCacheJournalTestPeer::resident(journal), 1u);
}

TEST(PrefixCacheJournalTest, ConcurrentPublishKeepsUniqueMonotonicSequence) {
  PrefixCacheJournal journal;
  constexpr uint64_t kThreads = 8;
  constexpr uint64_t kEventsPerThread = 100;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (uint64_t thread_id = 0; thread_id < kThreads; ++thread_id) {
    threads.emplace_back([thread_id, &journal]() {
      for (uint64_t index = 0; index < kEventsPerThread; ++index) {
        const uint64_t value = thread_id * kEventsPerThread + index + 1;
        PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(value));
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  const PrefixCacheEvents committed =
      PrefixCacheJournalTestPeer::events(journal, kThreads * kEventsPerThread);
  const size_t committed_pending = PrefixCacheJournalTestPeer::pending(journal);
  const size_t committed_resident =
      PrefixCacheJournalTestPeer::resident(journal);
  EXPECT_GT(committed.commit_version, 0u);
  EXPECT_EQ(committed_pending, 0u);
  EXPECT_EQ(committed_resident, kThreads * kEventsPerThread);
}

TEST(PrefixCacheJournalTest,
     LargeSnapshotRootIsStableDuringConcurrentMutation) {
  constexpr uint64_t kResidentEntries = 32768;
  PrefixCacheJournal journal;
  std::set<std::string> expected_entries;
  for (uint64_t index = 0; index < kResidentEntries; ++index) {
    const XXH3Key hash = make_hash(index + 1);
    expected_entries.insert(hash_key(hash));
    PrefixCacheJournalTestPeer::publish_stored(journal, hash);
  }
  const PrefixCacheEvents drained =
      PrefixCacheJournalTestPeer::events(journal, kResidentEntries);

  PrefixCacheSnapshot snapshot;
  std::atomic<bool> snapshot_started = false;
  std::thread snapshot_thread([&]() {
    snapshot_started.store(true, std::memory_order_release);
    snapshot = PrefixCacheJournalTestPeer::snapshot(journal);
  });

  while (!snapshot_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  uint64_t mutations = 0;
  for (uint64_t index = 0; index < 1024; ++index) {
    PrefixCacheJournalTestPeer::publish_removed(journal, make_hash(index + 1));
    PrefixCacheJournalTestPeer::publish_stored(journal,
                                               make_hash(100000 + index));
    ++mutations;
  }
  snapshot_thread.join();

  EXPECT_EQ(snapshot.commit_version, drained.commit_version);
  std::set<std::string> actual_entries;
  for (const XXH3Key& hash : snapshot.hashes) {
    actual_entries.insert(hash_key(hash));
  }
  const size_t resident_entries = PrefixCacheJournalTestPeer::resident(journal);
  EXPECT_EQ(snapshot.hashes.size(), kResidentEntries);
  EXPECT_EQ(actual_entries, expected_entries);
  EXPECT_EQ(mutations, 1024u);
  EXPECT_EQ(resident_entries, kResidentEntries);
}

TEST(PrefixCacheJournalTest, BelowHighWatermarkHonorsRequestedBatch) {
  // Empty resident uses the 8192 floor. 50 pending events sit under 60%.
  PrefixCacheJournal journal;
  for (uint64_t index = 0; index < 50; ++index) {
    PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(index + 1));
  }

  const PrefixCacheEvents batch =
      PrefixCacheJournalTestPeer::events(journal, /*max_events=*/8);
  EXPECT_EQ(batch.events.size(), 8u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::pending(journal), 42u);
}

TEST(PrefixCacheJournalTest, EachExtraCapacityBandAddsOneTenth) {
  // 9000 resident hashes raise the capacity above the 8192 floor. 60% of
  // 9000 is 5400, and each band is 900. 6300 pending events are one band
  // above that line, so this call returns the requested 8 plus 900.
  constexpr size_t kResident = 9000;
  constexpr size_t kPending = 6300;
  PrefixCacheJournal journal;
  for (uint64_t index = 0; index < kResident; ++index) {
    PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(index + 1));
  }
  ASSERT_EQ(
      PrefixCacheJournalTestPeer::events(journal, kResident).events.size(),
      kResident);
  for (uint64_t index = 0; index < kPending; ++index) {
    PrefixCacheJournalTestPeer::publish_stored(
        journal, make_hash(kResident + index + 1));
  }

  const PrefixCacheEvents batch =
      PrefixCacheJournalTestPeer::events(journal, /*max_events=*/8);
  EXPECT_EQ(batch.events.size(), 908u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::pending(journal), 5392u);
}

TEST(PrefixCacheJournalTest, FullQueueDrainsBackTowardHighWatermark) {
  // Capacity follows the 9000 resident hashes. A queue as deep as that
  // capacity sits 3600 events above the 5400 line, four bands of 900. One
  // call returns the requested 8 plus those 3600.
  constexpr size_t kResident = 9000;
  PrefixCacheJournal journal;
  for (uint64_t index = 0; index < kResident; ++index) {
    PrefixCacheJournalTestPeer::publish_stored(journal, make_hash(index + 1));
  }
  ASSERT_EQ(
      PrefixCacheJournalTestPeer::events(journal, kResident).events.size(),
      kResident);
  for (uint64_t index = 0; index < kResident; ++index) {
    PrefixCacheJournalTestPeer::publish_stored(
        journal, make_hash(kResident + index + 1));
  }
  EXPECT_EQ(PrefixCacheJournalTestPeer::pending(journal), kResident);

  const PrefixCacheEvents batch =
      PrefixCacheJournalTestPeer::events(journal, /*max_events=*/8);
  EXPECT_EQ(batch.events.size(), 3608u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::pending(journal), 5392u);

  // Back under the line, the next call is the caller's batch again.
  const PrefixCacheEvents steady =
      PrefixCacheJournalTestPeer::events(journal, /*max_events=*/8);
  EXPECT_EQ(steady.events.size(), 8u);
  EXPECT_EQ(PrefixCacheJournalTestPeer::pending(journal), 5384u);
}

}  // namespace
}  // namespace xllm
