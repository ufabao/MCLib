#pragma once
#include <condition_variable>
#include <queue>
#include <mutex>
using namespace std;

template <class T>
class ConcurrentQueue
{

  queue<T> myQueue;
	mutable mutex myMutex;
	condition_variable myCV;
	bool myInterrupt;

public:

	ConcurrentQueue() : myInterrupt(false) {}
	~ConcurrentQueue() { interrupt(); }

	bool empty() const
	{
		lock_guard<mutex> lk(myMutex);
		return myQueue.empty();
	}	

	bool tryPop(T& t)
	{
		lock_guard<mutex> lk(myMutex);
		if (myQueue.empty()) return false;
		t = move(myQueue.front());
		myQueue.pop();

		return true;
	}

	void push(T t)
	{
		{
			lock_guard<mutex> lk(myMutex);
			myQueue.push(move(t));
		}	

		myCV.notify_one();
	}

	bool pop(T& t)
	{
		unique_lock<mutex> lk(myMutex);

		while (!myInterrupt && myQueue.empty()) myCV.wait(lk);

		if (myInterrupt) return false;

		t = move(myQueue.front());
		myQueue.pop();

		return true;

	}

	void interrupt()
	{
        {
            lock_guard<mutex> lk(myMutex);
            myInterrupt = true;
        }
		myCV.notify_all();
	}

    void resetInterrupt()
    {
        myInterrupt = false;
    }

    void clear()
    {
        queue<T> empty;
        swap(myQueue, empty);
    }
};