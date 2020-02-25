
# two sum
# 给定一个整数数组 nums 和一个目标值 target，
# 请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
# 你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

# 暴力方法
from typing import List, Tuple, Dict

class Solution_1:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for idx, each in enumerate(nums):
            query = each
            for ids, search in enumerate(nums):
                if idx == ids:
                    continue
                else:
                    if query + search == target:
                        list_i = sorted([ids, idx])
                        return list_i




# 回首查找
# 每一个元素只查找之前的元素
class Solution_2:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for idx, each in enumerate(nums):
            for ids, ele in enumerate(nums):
                if ids >= idx:
                    break
                else:
                    if each + ele == target:
                        
                        return sorted([ids, idx])


