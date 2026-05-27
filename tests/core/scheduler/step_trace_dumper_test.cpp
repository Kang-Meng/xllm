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

#include "scheduler/step_trace_dumper.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

namespace xllm {

TEST(StepTraceDumperTest, FormatsTsvRecord) {
  StepTraceRecord record;
  record.ts_us = 123456;
  record.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  record.qlen = 1024;
  record.step_latency_ms = 12.3456;

  const std::string line = StepTraceDumper::format_record_tsv(record);
  EXPECT_EQ(line, "123456\tprefill\t1024\t12.346\n");
}

TEST(StepTraceDumperTest, DumpsRecordsAsynchronously) {
  char tmp_path[] = "/tmp/xllm_step_trace_test_XXXXXX";
  const int fd = mkstemp(tmp_path);
  ASSERT_GE(fd, 0);
  close(fd);

  {
    StepTraceDumper dumper(tmp_path, 8, 1, 1);
    StepTraceRecord record;
    record.ts_us = 789;
    record.batch_forward_type = BatchForwardType::DECODE;
    record.qlen = 4;
    record.step_latency_ms = 3.25;
    EXPECT_TRUE(dumper.try_enqueue(record));
  }

  std::ifstream input(tmp_path);
  ASSERT_TRUE(input.is_open());
  const std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("ts_us\tstep_kind\tqlen\tstep_latency_ms\n"),
            std::string::npos);
  EXPECT_NE(content.find("789\tdecode\t4\t3.250\n"), std::string::npos);

  std::remove(tmp_path);
}

}  // namespace xllm
