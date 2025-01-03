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
		int result=1;
		while(t-->0)
		{
		    int a=input.nextInt();
		    int b=input.nextInt();
		    int c=(a+5)/6;
		   
		     result=c*b;
		   
		    System.out.println(result);
		}

	}
}
