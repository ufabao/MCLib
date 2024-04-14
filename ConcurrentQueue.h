#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>




template <class T> class ConcurrentQueue {

  std::queue<T> myQueue;
  mutable std::mutex myMutex;
  std::condition_variable myCV;
  bool myInterrupt;

public:
  ConcurrentQueue() : myInterrupt(false) {}
  ~ConcurrentQueue() { interrupt(); }

  bool empty() const {
    std::lock_guard<std::mutex> lk(myMutex);
    return myQueue.empty();
  }

  bool tryPop(T &t) {
    std::lock_guard<std::mutex> lk(myMutex);
    if (myQueue.empty())
      return false;
    t = move(myQueue.front());
    myQueue.pop();

    return true;
  }

  void push(T t) {
    {
      std::lock_guard<std::mutex> lk(myMutex);
      myQueue.push(move(t));
    }

    myCV.notify_one();
  }

  bool pop(T &t) {
    std::unique_lock<std::mutex> lk(myMutex);

    while (!myInterrupt && myQueue.empty())
      myCV.wait(lk);

    if (myInterrupt)
      return false;

    t = move(myQueue.front());
    myQueue.pop();

    return true;
  }

  void interrupt() {
    {
      std::lock_guard<std::mutex> lk(myMutex);
      myInterrupt = true;
    }
    myCV.notify_all();
  }

  void resetInterrupt() { myInterrupt = false; }

  void clear() {
    std::queue<T> empty;
    swap(myQueue, empty);
  }
};