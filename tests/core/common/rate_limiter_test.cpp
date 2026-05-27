#include "rate_limiter.h"

#include <gtest/gtest.h>

#include "global_flags.h"

namespace xllm {

TEST(RequestLimiterTest, Basic) {
  // Set the maximum number of concurrent requests to 1.
  FLAGS_max_concurrent_requests = 1;
  RateLimiter rate_limiter;
  // The current number of concurrent requests is 0, no rate limiting is
  // applied.
  EXPECT_EQ(rate_limiter.is_limited(), false);
  // The current number of concurrent requests is 1, rate limiting is applied.
  EXPECT_EQ(rate_limiter.is_limited(), true);
  // Decrease the number of concurrent requests by one, changing the concurrency
  // from 1 to 0.
  rate_limiter.decrease_one_request();
  // The current number of concurrent requests is 0, no rate limiting is
  // applied.
  EXPECT_EQ(rate_limiter.is_limited(), false);
}

TEST(RequestLimiterTest, RequestGuardReleasesAcquiredRequestOnScopeExit) {
  FLAGS_max_concurrent_requests = 1;
  RateLimiter rate_limiter;

  ASSERT_FALSE(rate_limiter.is_limited());
  {
    auto guard = rate_limiter.make_request_guard();
    EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  }

  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
  EXPECT_FALSE(rate_limiter.is_limited());
}

TEST(RequestLimiterTest, DismissedRequestGuardTransfersReleaseResponsibility) {
  FLAGS_max_concurrent_requests = 1;
  RateLimiter rate_limiter;

  ASSERT_FALSE(rate_limiter.is_limited());
  {
    auto guard = rate_limiter.make_request_guard();
    guard.dismiss();
  }

  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  rate_limiter.decrease_one_request();
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
}

}  // namespace xllm
