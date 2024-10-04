# Python

## Data Types
### Text
- str
### Numeric
- int
- float
- complex
### Sequence
### Mapping

### Set
- set
- frozenset
### Boolean
- bool
### Binary
- byte
- byrtearray
- memoryview

## Conditions
```
if con1:
    operation1
elif cond2:
    operation2
else:
    operation3
```

## Loops
for i in range(5):
    if i == 3:
        break
else:
    print('break was called')

## Functions

## File Handling

## Advanced Data Types
### Heap
```
import heapq
heapq.heapify(list)
heapq.heappop
heapq.heappush()
```
### Deque
```
from collections import dequq

q = deque()
q.append(x)
q.appendleft(x)
q.pop()
q.popleft(e)
q.rotate()
```

## String operations

### Slicing String
```
s[i:j] -> chars i to j excluded
s[i:] -> chars i to end
s[:j] -> chars from start to j excluded
s[:] -> copy string
```

### Format strings
```
f"Hello {country}"
```

### Multiple copies
```
s = '123'
s * 3 # 123123123
```

### Join strings with delimiter
```
words = ['Alice','Bob', 'Charles']
",".join(words) # Alice,Bob,Charles 
```

