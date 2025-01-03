import java.util.*;
import java.lang.*;
import java.io.*;

class Codechef
{
	public static void main (String[] args) throws java.lang.Exception
	{
		// your code goes here
    Scanner input=new Scanner(System.in);
    int t=input.nextInt();
    while(t-->0)
    {
        int a =input.nextInt();
        if(Math.abs(a-10)>=3)
        System.out.println("Yes");
        else
        System.out.println("No");
    }
	}
}
