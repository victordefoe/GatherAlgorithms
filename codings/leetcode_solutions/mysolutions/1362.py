
# 最接近的因数

# 给你一个整数 num，请你找出同时满足下面全部要求的两个整数：

#     两数乘积等于  num + 1 或 num + 2
#     以绝对差进行度量，两数大小最接近

# 你可以按任意顺序返回这两个整数

# example

# 输入：num = 8
# 输出：[3,3]
# 解释：对于 num + 1 = 9，
# 最接近的两个因数是 3 & 3；对于 num + 2 = 10, 最接近的两个因数是 2 & 5，因此返回 3 & 3 。

from typing import List, Tuple, Dict

# overtime solution... 
class Solution:
    def closestDivisors(self, num: int) -> List[int]:
        import numpy as np
        import math
        from itertools import permutations, combinations, combinations_with_replacement
    
        m = [[],[]]
        for idx, n in enumerate([num+1, num+2]):
            while n!=1:    #n==1时，已分解到最后一个质因数
                for i in range(2,int(n+1)):
                    if n % i == 0:
                        m[idx].append(i)    #将i转化为字符串再追加到列表中，便于使用join函数进行输出
                        n = n/i
                if n==1:
                    m[idx].append(1)
                    break    #n==1时，循环停止
        print(m)
        
        
        
        # find out all combinations in list
        pairs = []
        for idx, h in enumerate(m):
            for n_token in range(math.ceil(len(h)/2)):
                for arr in combinations_with_replacement(h, n_token):
                    
                    a = np.prod(arr)
                    b = [num+1, num+2][idx] / a
                    pairs.append([int(a),int(b)])
        print(pairs)
        mat = np.array(pairs).transpose()
        return pairs[np.argmin(np.abs(mat[0]-mat[1]))]
