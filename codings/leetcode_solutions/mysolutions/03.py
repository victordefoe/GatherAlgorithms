

# 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度

# example

# 输入: "abcabcbb"
# 输出: 3 
# 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。


# 滑动窗口 O(N)
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        no_repeat_list = []
        best_length = 0
        for char in s: 
            if char in no_repeat_list:
                best_length = max([len(no_repeat_list), best_length])
                no_repeat_list = no_repeat_list[no_repeat_list.index(char)+1:]
                no_repeat_list.append(char)
            else:
                no_repeat_list.append(char)
        best_length = max([len(no_repeat_list), best_length])
        return best_length
            