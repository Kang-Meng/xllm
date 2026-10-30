/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/util/net.h"

#include <gtest/gtest.h>

namespace xllm {
namespace net {
namespace {

TEST(NetTest, SelectsLoopbackForLoopbackRemote) {
  EXPECT_EQ(get_route_ip("127.0.0.1:19888"), "127.0.0.1");
}

TEST(NetTest, ResolvesRemoteHostname) {
  EXPECT_EQ(get_route_ip("localhost:19888"), "127.0.0.1");
}

TEST(NetTest, NormalizesLocalBindHosts) {
  const std::string local_ip = get_local_ip_addr();
  ASSERT_FALSE(local_ip.empty());

  EXPECT_EQ(extract_ip("0.0.0.0"), local_ip);
  EXPECT_EQ(extract_ip("127.0.0.1"), local_ip);
  EXPECT_EQ(extract_ip("localhost"), local_ip);

  const std::string wildcard_endpoint = "0.0.0.0:19000";
  EXPECT_EQ(
      extract_ip(wildcard_endpoint) + ":" + extract_port(wildcard_endpoint),
      local_ip + ":19000");
}

TEST(NetTest, ParsesBracketedIpv6Endpoint) {
  std::string host;
  int port = 0;
  parse_host_port_from_addr("[2001:db8::1]:35500", host, port);
  EXPECT_EQ(host, "2001:db8::1");
  EXPECT_EQ(port, 35500);
}

TEST(NetTest, FormatsAndValidatesIpv4HostnameAndIpv6Endpoints) {
  EXPECT_EQ(format_host_port("192.0.2.10", 19000), "192.0.2.10:19000");
  EXPECT_EQ(format_host_port("engine.internal", 19000),
            "engine.internal:19000");
  EXPECT_EQ(format_host_port("2001:db8::10", 19000), "[2001:db8::10]:19000");
  EXPECT_EQ(format_host_port("[2001:db8::10]", 19000), "[2001:db8::10]:19000");
  EXPECT_TRUE(is_valid_host_port("192.0.2.10:19000"));
  EXPECT_TRUE(is_valid_host_port("engine.internal:19000"));
  EXPECT_TRUE(is_valid_host_port("[2001:db8::10]:19000"));
  EXPECT_FALSE(is_valid_host_port("2001:db8::10:19000"));
  EXPECT_FALSE(is_valid_host_port("[2001:db8::10]"));
  EXPECT_FALSE(is_valid_host_port("[engine.internal]:19000"));
  EXPECT_FALSE(is_valid_host_port("bad host:19000"));
  EXPECT_FALSE(is_valid_host_port("[fe80::1%eth0]:19000"));
  EXPECT_TRUE(format_host_port("fe80::1%eth0", 19000).empty());
  EXPECT_TRUE(format_host_port("[fe80::1%eth0]", 19000).empty());
  EXPECT_FALSE(is_valid_host_port("engine.internal:0"));
  EXPECT_TRUE(format_host_port("", 19000).empty());
  EXPECT_TRUE(format_host_port("[2001:db8::10", 19000).empty());
  EXPECT_TRUE(format_host_port("engine.internal:19000", 19000).empty());
  EXPECT_TRUE(format_host_port("[engine.internal]", 19000).empty());
  EXPECT_EQ(format_host_port("127.0.0.1", 0), "127.0.0.1:0");
  EXPECT_TRUE(format_host_port("127.0.0.1", 65536).empty());
}

TEST(NetTest, ReturnsEmptyWhenRemoteHasNoUsableRoute) {
  EXPECT_TRUE(get_route_ip("255.255.255.255:19888").empty());
}

TEST(NetTest, ReturnsEmptyWhenRemoteCannotBeResolved) {
  EXPECT_TRUE(get_route_ip("worker.invalid:19888").empty());
}

}  // namespace
}  // namespace net
}  // namespace xllm
