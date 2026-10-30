/* Copyright 2025-2026 The xLLM Authors.

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

#include "net.h"

#include <arpa/inet.h>
#include <glog/logging.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_set>

namespace xllm {
namespace net {

namespace {

std::mutex g_port_mutex;
std::unordered_set<int> g_allocated_port_map;

std::string to_ip_addr(const sockaddr_storage& storage) {
  char ip[INET6_ADDRSTRLEN]{'\0'};
  const void* address = nullptr;
  if (storage.ss_family == AF_INET) {
    address = &reinterpret_cast<const sockaddr_in*>(&storage)->sin_addr;
  } else if (storage.ss_family == AF_INET6) {
    address = &reinterpret_cast<const sockaddr_in6*>(&storage)->sin6_addr;
  } else {
    return "";
  }
  const char* result = inet_ntop(storage.ss_family, address, ip, sizeof(ip));
  if (result == nullptr) {
    return "";
  }
  return std::string(ip);
}

struct HostPort {
  std::string host;
  uint16_t port = 0;
};

bool valid_unbracketed_host(std::string_view host) {
  return !host.empty() &&
         std::none_of(host.begin(), host.end(), [](unsigned char character) {
           return character <= 0x20 || character == 0x7f || character == '[' ||
                  character == ']' || character == ':' || character == '/';
         });
}

bool valid_ipv6_host(std::string_view host) {
  // Scoped addresses are interface-local and produce platform-dependent
  // inet_pton results. They cannot be advertised safely across Engine hosts.
  if (host.find('%') != std::string_view::npos) {
    return false;
  }
  const std::string text(host);
  in6_addr parsed{};
  return inet_pton(AF_INET6, text.c_str(), &parsed) == 1;
}

std::optional<HostPort> try_parse_host_port(std::string_view address) {
  if (address.empty()) {
    return std::nullopt;
  }
  std::string_view host;
  std::string_view port_text;
  if (address.front() == '[') {
    const size_t closing_bracket = address.find(']');
    if (closing_bracket == std::string_view::npos || closing_bracket == 1 ||
        closing_bracket + 2 > address.size() ||
        address[closing_bracket + 1] != ':') {
      return std::nullopt;
    }
    host = address.substr(1, closing_bracket - 1);
    port_text = address.substr(closing_bracket + 2);
    if (!valid_ipv6_host(host)) {
      return std::nullopt;
    }
  } else {
    const size_t separator = address.rfind(':');
    if (separator == std::string_view::npos || separator == 0 ||
        separator + 1 >= address.size() || address.find(':') != separator) {
      return std::nullopt;
    }
    host = address.substr(0, separator);
    port_text = address.substr(separator + 1);
    if (!valid_unbracketed_host(host)) {
      return std::nullopt;
    }
  }

  uint32_t port = 0;
  const auto parsed = std::from_chars(
      port_text.data(), port_text.data() + port_text.size(), port);
  if (parsed.ec != std::errc() ||
      parsed.ptr != port_text.data() + port_text.size() || port == 0 ||
      port > std::numeric_limits<uint16_t>::max()) {
    return std::nullopt;
  }
  return HostPort{.host = std::string(host),
                  .port = static_cast<uint16_t>(port)};
}

}  // namespace

std::string format_host_port(const std::string& host, int32_t port) {
  if (host.empty() || port < 0 || port > std::numeric_limits<uint16_t>::max()) {
    return "";
  }
  const bool bracketed = host.front() == '[';
  std::string normalized_host = host;
  if (bracketed) {
    if (host.size() < 3 || host.back() != ']') {
      return "";
    }
    normalized_host = host.substr(1, host.size() - 2);
  } else if (host.find('[') != std::string::npos ||
             host.find(']') != std::string::npos) {
    return "";
  }
  const bool is_ipv6 = normalized_host.find(':') != std::string::npos;
  if ((bracketed && !is_ipv6) ||
      (is_ipv6 && !valid_ipv6_host(normalized_host)) ||
      (!is_ipv6 && !valid_unbracketed_host(normalized_host))) {
    return "";
  }
  return (is_ipv6 ? "[" + normalized_host + "]" : normalized_host) + ":" +
         std::to_string(port);
}

bool is_valid_host_port(const std::string& address) {
  return try_parse_host_port(address).has_value();
}

std::string get_local_ip_addr() {
  char ip[INET_ADDRSTRLEN]{'\0'};
  char hostname[256];
  int ret = gethostname(hostname, sizeof(hostname));
  if (ret != 0) {
    LOG(ERROR) << "gethostname failed";
    return "";
  }
  struct addrinfo* info = nullptr;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  ret = getaddrinfo(hostname, nullptr, &hints, &info);
  if (ret != 0) {
    LOG(ERROR) << "getaddrinfo failed";
    return "";
  }
  std::unique_ptr<struct addrinfo, decltype(&freeaddrinfo)> guard(info,
                                                                  freeaddrinfo);
  const sockaddr_in* addr = reinterpret_cast<const sockaddr_in*>(info->ai_addr);
  const char* result =
      inet_ntop(addr->sin_family, &addr->sin_addr, ip, sizeof(ip));
  if (result == nullptr) {
    LOG(ERROR) << "inet_ntop failed";
    return "";
  }

  return std::string(ip);
}

std::string get_route_ip(const std::string& remote_addr) {
  std::string remote_host;
  int remote_port = 0;
  parse_host_port_from_addr(remote_addr, remote_host, remote_port);

  struct addrinfo* info = nullptr;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  std::string port = std::to_string(remote_port);
  int ret = getaddrinfo(remote_host.c_str(), port.c_str(), &hints, &info);
  if (ret != 0) {
    LOG(ERROR) << "Failed to resolve remote address " << remote_addr << ": "
               << gai_strerror(ret);
    return "";
  }
  std::unique_ptr<struct addrinfo, decltype(&freeaddrinfo)> guard(info,
                                                                  freeaddrinfo);

  for (const struct addrinfo* current = info; current != nullptr;
       current = current->ai_next) {
    const int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
      continue;
    }
    ret = connect(fd, current->ai_addr, current->ai_addrlen);
    if (ret != 0) {
      ::close(fd);
      continue;
    }

    sockaddr_in local{};
    socklen_t local_len = sizeof(local);
    ret =
        getsockname(fd, reinterpret_cast<struct sockaddr*>(&local), &local_len);
    ::close(fd);
    if (ret != 0) {
      continue;
    }

    sockaddr_storage local_storage{};
    std::memcpy(&local_storage, &local, sizeof(local));
    std::string local_ip = to_ip_addr(local_storage);
    if (!local_ip.empty()) {
      return local_ip;
    }
  }

  LOG(ERROR) << "No local route found for remote address " << remote_addr;
  return "";
}

int get_local_free_port() {
  std::lock_guard<std::mutex> lock(g_port_mutex);
  int port = 0;
  do {
    port = 0;
    struct sockaddr_in addr;
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return -1;
    }
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
      ::close(fd);
      return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &len) ==
        -1) {
      ::close(fd);
      return -1;
    }
    port = ntohs(addr.sin_port);

    ::close(fd);
  } while (g_allocated_port_map.find(port) != g_allocated_port_map.end());

  g_allocated_port_map.insert(port);

  return port;
}

uint64_t convert_ip_port_to_uint64(const std::string& ip, uint16_t port) {
  in_addr ip_addr;
  CHECK(inet_pton(AF_INET, ip.c_str(), &ip_addr) == 1)
      << "Invalid IPv4 address format : " << ip;

  uint32_t ip_network = ip_addr.s_addr;
  return (static_cast<uint64_t>(ip_network) << 32) | port;
}

std::pair<std::string, uint16_t> convert_uint64_to_ip_port(uint64_t input) {
  uint16_t port = static_cast<uint16_t>(input & 0xFFFF);
  uint32_t ip_network = static_cast<uint32_t>(input >> 32);

  in_addr ip_addr;
  ip_addr.s_addr = ip_network;

  char ip_str[INET_ADDRSTRLEN];
  const char* result = inet_ntop(AF_INET, &ip_addr, ip_str, INET_ADDRSTRLEN);
  CHECK(result != nullptr) << "Failed to convert IP address from uint64: "
                           << input;

  return {std::string(ip_str), port};
}

// input example: 127.0.0.1:18889
std::string extract_ip(const std::string& input) {
  const std::optional<HostPort> endpoint = try_parse_host_port(input);
  std::string ip = endpoint.has_value() ? endpoint->host : input;
  if (ip == "127.0.0.1" || ip == "0.0.0.0" || ip == "localhost") {
    ip = get_local_ip_addr();
  }
  return ip;
}

std::string extract_port(const std::string& input) {
  const std::optional<HostPort> endpoint = try_parse_host_port(input);
  return endpoint.has_value() ? std::to_string(endpoint->port) : "";
}

void parse_host_port_from_addr(const std::string& addr,
                               std::string& host,
                               int& port) {
  const std::optional<HostPort> endpoint = try_parse_host_port(addr);
  CHECK(endpoint.has_value()) << "Invalid address format: " << addr;
  host = endpoint->host;
  port = endpoint->port;
}

}  // namespace net
}  // namespace xllm
