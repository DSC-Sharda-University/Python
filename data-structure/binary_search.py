""" in this programme we are going to implement binary search algorithm  using python"""

def binary_search(arr,target):
    high = len(arr)
    low = 0
    while(low <= high):
        mid = (low + high)//2
        if(target == arr[mid]) :
            return arr[mid]
        elif(arr[mid]< target):
                high = mid-1
        else:
            low = mid +1
    return False
arr = list(range(20))
#target = 10
#print(binary_search(arr,target))

arr=[]
n = int(input("Enter the number of element in your array:\n"))
for i in range(n):
    arr.append(int(input("Enter an element into the array:")))
target =  int(input("Enter the target :\n"))
""" here we are trying to make sure that the array is sorted otherwise our algorithm     cannot work"""
print(binary_search(arr,target))

