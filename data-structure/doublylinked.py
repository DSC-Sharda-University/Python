"""
To create doubly link list with the operation insert at the first location and
also insert at the last location.
"""


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class doublylink:
    def __init__(self):
        self.head = None

    def insert_last(self, data):
        if self.head is None:
            node = Node(data)
            node.prev = None
            self.head = node
        else:
            node = Node(data)
            temp = self.head
            while temp.next:
                temp = temp.next
            temp.next = node
            node.prev = temp
            node.next = None

    def insert_first(self, data):
        if self.head is None:
            node = Node(data)
            node.prev = None
            self.head = node
        else:
            node = Node(data)
            self.head.prev = node
            node.next = self.head
            self.head = node
            node.prev = None

    def display(self):
        temp = self.head
        while temp:
            print(temp.data, '==>', end='')
            temp = temp.next


link = doublylink()
while(True):
    print("1-Insert first\n 2-Insert last\n 3-Display\n")
    choice = int(input("Enter your choice:\n"))
    if choice == 1:
        link.insert_first(int(input(" \nEnter the value to insert:\n")))
    elif choice == 2:
        link.insert_last(int(input("Enter the value to insert at the last\
                                   postion:\n")))
    elif choice == 3:
        link.display()
