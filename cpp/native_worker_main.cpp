#include "calmetrics_engine/native_process.hpp"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "CalMetricsEngine native-worker protocol=1 python_runtime=0\n";
    return 0;
  }
  if (argc != 2 || std::string(argv[1]) != "--stdio")
    return 2;
  try {
    return calmetrics_engine::native::worker_main();
  } catch (...) {
    return 2;
  }
}
