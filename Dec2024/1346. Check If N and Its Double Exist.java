class Solution {
    public boolean checkIfExist(int[] arr) {
   Map<Integer,Integer>mp=new HashMap<>();
   for(int i=0;i<arr.length;i++)
   {
    mp.put(arr[i],mp.getOrDefault(arr[i],0)+1);
   }

   for(int j=0;j<arr.length;j++)
   {
    if(arr[j]!=0 && mp.containsKey(2*arr[j]))
    return true;
    else if(arr[j]==0&&mp.get(arr[j])>1)
    return true;
   }
   return false;
    }
}
