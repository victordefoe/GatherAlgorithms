

# 给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。

# 请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

# 你可以假设 nums1 和 nums2 不会同时为空。

# example:
# nums1 = [1, 3]
# nums2 = [2]

# 则中位数是 2.0

# 来源：力扣（LeetCode）
# 链接：https://leetcode-cn.com/problems/median-of-two-sorted-arrays
# 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

from typing import List, Tuple, Dict
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums = nums1 + nums2
        nums.sort()
        
        location = len(nums)
        if location % 2 == 0:
            media = 0.5 * ( nums[int(location/2)-1] + nums[int(location/2)] )
        else:
            media = nums[int(location/2)]
        return media