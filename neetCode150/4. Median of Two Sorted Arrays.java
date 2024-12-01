class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n=nums1.length;
        int m=nums2.length;
        int i=0,j=0,k=0;
        int []merged=new int [n+m];
        double sum=0;
        while(i<n&&j<m)
        {
            if(nums1[i]<nums2[j])
            {
                merged[k]=nums1[i];
                i++;
                k++;
            }
            else
            {
                merged[k]=nums2[j];
                j++;
                k++;
            }
        
        }
        while(i<n)
        {
            merged[k]=nums1[i];
                i++;
                k++;
        }
        while(j<m&&k<n+m)
        {
            merged[k]=nums2[j];
                j++;
                k++;
        }

        if(merged.length%2!=0)
        return merged[merged.length/2];
        else
        {
            int a =merged.length;
            return (merged[a / 2 - 1] + merged[a / 2]) / 2.0;

        }
      
    }
}
