class Solution {
    public int climbStairs(int n) {
        int dp[]=new int[n+1];
        return helper(0,n,dp);
    }
    public int helper(int i ,int n,int[]dp)
    {

        if(i==n)
        return 1;
        if(i>n)
        return 0;
        if(dp[i]>0)
        {
            return dp[i];
        }
        dp[i]=helper(i+1,n,dp)+helper(i+2,n,dp);
        return dp[i];
    }
}


class Solution {
    public int climbStairs(int n) {
        int []dp=new int [n+1];
            if(n==1)
            return 1;
        dp[1]=1;
        dp[2]=2;
        for(int i=3;i<=n;i++)
        {
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n];
    }
}
