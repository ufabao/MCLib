#pragma once
#include "ConcurrentQueue.h"
#include <functional>
#include <future>
#include <thread>

using namespace std::chrono_literals;

using Task =  std::packaged_task<bool(void)>;
using TaskHandle = std::future<bool>;

class ThreadPool {
  static ThreadPool myInstance;
  ConcurrentQueue<Task> myQueue;
  std::vector<std::thread> myThreads;
  bool myActive;
  bool myInterrupt;
  static thread_local size_t myTLSNum;

  void threadFunc(const size_t num) {
    myTLSNum = num;

    Task t;
    while (!myInterrupt) {
      myQueue.pop(t);
      if (!myInterrupt)
        t();
    }
  }

  ThreadPool() : myActive(false), myInterrupt(false) {}

public:
  static ThreadPool *getInstance() { return &myInstance; }

  size_t numThreads() const { return myThreads.size(); }

  static size_t threadNum() { return myTLSNum; }

  void start(const size_t nThread = std::thread::hardware_concurrency() - 1) {
    if (!myActive) {
      myThreads.reserve(nThread);

      for (size_t i = 0; i < nThread; i++)
        myThreads.push_back(std::thread(&ThreadPool::threadFunc, this, i + 1));

      myActive = true;
    }
  }

  ~ThreadPool() { stop(); }

  void stop() {
    if (myActive) {
      myInterrupt = true;

      myQueue.interrupt();

      for_each(myThreads.begin(), myThreads.end(), mem_fn(&std::thread::join));

      myThreads.clear();

      myQueue.clear();
      myQueue.resetInterrupt();

      myActive = false;

      myInterrupt = false;
    }
  }

  ThreadPool(const ThreadPool &rhs) = delete;
  ThreadPool &operator=(const ThreadPool &rhs) = delete;
  ThreadPool(ThreadPool &&rhs) = delete;
  ThreadPool &operator=(ThreadPool &&rhs) = delete;

  template <typename Callable> TaskHandle spawnTask(Callable c) {
    Task t(std::move(c));
    TaskHandle f = t.get_future();
    myQueue.push(std::move(t));
    return f;
  }

  bool activeWait(const TaskHandle &f) {
    Task t;
    bool b = false;

    while (f.wait_for(0s) != std::future_status::ready) {
      if (myQueue.tryPop(t)) {
        t();
        b = true;
      } else {
        f.wait();
      }
    }

    return b;
  }
};