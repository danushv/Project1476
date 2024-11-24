class Solution {
    public int[] twoSum(int[] nums, int target) {
        int result[]=new int[2];
        int n=nums.length;
        for(int i=0;i<n-1;i++)
        {
            for(int j=i+1;j<n;j++)
            {
                if(nums[i]+nums[j]==target)
                {
                result[0]=i;
                result[1]=j;
                }
            }
        }
        return result;
    }
}
// The above is O(n^2) and we can do it one pass below is the explaination 


class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer,Integer> map=new HashMap<>();
        int n=nums.length;
        for(int i=0;i<n;i++)
        {
            int complement=target-nums[i];
            if(map.containsKey(complement))
            return new int[]{map.get(complement),i};

            map.put(nums[i],i);
        }
     return new int []{};
    }
}
