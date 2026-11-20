import java.util.Random;
public class RngFixture {public static void main(String[]a){Random r=new Random(190513L);int[] bounds={256,64,256,6,3,2147483647};for(int i=0;i<10000;i++)System.out.println(r.nextInt(bounds[i%bounds.length]));}}
