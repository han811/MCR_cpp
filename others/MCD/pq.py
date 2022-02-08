import heapq
class PriorityQueue:
    def __init__(self):
        self.pq = []
    
    def push(self, priority, value):
        entry = [priority, value]
        heapq.heappush(self.pq, entry)
        
    def pop(self):
        if self.pq:
            priority, value = heapq.heappop(self.pq)
            return priority, value
        else:
            return None
    
    def size(self):
        return len(self.pq)