
# 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。

# 如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

# 您可以假设除了数字 0 之外，这两个数都不会以 0 开头。



from typing import List, Tuple, Dict

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        res = ListNode(0)
        res.val = self.addition(l1.val, l2.val, 0)[1]
        go = res
        # for i in range(len(l1)):
        carry = self.addition(l1.val, l2.val, 0)[0]
        
        l1 = l1.next
        l2 = l2.next
        for i in range(int(1e5)):
            
            if l1 is not None:
                fst_num = l1.val
                l1 = l1.next
            else:
                fst_num = 0
            if l2 is not None:
                sec_num = l2.val
                l2 = l2.next
            else:
                sec_num = 0
            
            
            temp_sum = self.addition(fst_num, sec_num, carry)
            val = temp_sum[1]
            
                   
            carry = temp_sum[0]

            if l1 is None and l2 is None and val+carry==0:
                break

            go.next = ListNode(val)
            go = go.next
            
        return res


    def addition(self, a, b, c):
        digit = (a + b + c) % 10
        carry = (a + b + c) // 10

        return tuple([carry, digit])

