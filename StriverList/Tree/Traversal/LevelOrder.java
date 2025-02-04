/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> q=new LinkedList<TreeNode> ();
        List<List<Integer>> result=new LinkedList<List<Integer>>();

        if(root==null)
        return result;
        q.offer(root);
        while(!q.isEmpty())
        {
            int length=q.size();
            List<Integer> sublist=new LinkedList<>();
            for(int i=0;i<length;i++)
            {
                if(q.peek().left!=null)
                q.offer(q.peek().left);
                if(q.peek().right!=null)
                q.offer(q.peek().right);
                sublist.add(q.poll().val);
            }
            result.add(sublist);
        }
        return result;
    }
}
