package org.lwjgl.opengl;
import java.nio.*;import java.util.*;
// Records the actual game's display-list construction; does not rasterize pixels.
public class GL11 {
 public static float[] projection=new float[16],modelview=new float[16];
 public static int frame,pass,current,vertices,frontVertices,nextList=1;
 public static double cx,cy,cz; public static FloatBuffer vertexData;
 public static List<int[]> compiled=new ArrayList<>();
 public static Map<Integer,Integer> sizes=new HashMap<>();
 public static Map<Integer,Integer> frontSizes=new HashMap<>();
 public static List<int[]> drawnFrontLists=new ArrayList<>();
 public static void camera(double x,double y,double z,double aspect){
  cx=x;cy=y;cz=z;Arrays.fill(projection,0);Arrays.fill(modelview,0);
  double f=1/Math.tan(Math.toRadians(35)),near=.05,far=1000;
  projection[0]=(float)(f/aspect);projection[5]=(float)f;projection[10]=(float)((far+near)/(near-far));projection[11]=-1;projection[14]=(float)(2*far*near/(near-far));
  modelview[0]=modelview[5]=modelview[10]=modelview[15]=1;
  modelview[12]=(float)-x;modelview[13]=(float)-y;modelview[14]=(float)-z;
 }
 public static void glGetFloat(int n,FloatBuffer b){b.put(n==2983?projection:modelview);}
 public static void glGenTextures(IntBuffer b){b.put(0,1);}
 public static int glGenLists(int n){int old=nextList;nextList+=n;return old;}
 public static void glNewList(int id,int mode){current=id;vertices=0;frontVertices=0;}
 public static void glEndList(){compiled.add(new int[]{frame,pass,current,vertices,frontVertices});sizes.put(current,vertices);frontSizes.put(current,frontVertices);}
 public static void glDrawArrays(int mode,int first,int count){
  vertices+=count;
  for(int i=first;i+3<first+count;i+=4){
   double x=vertexData.get(i*3),y=vertexData.get(i*3+1),z=vertexData.get(i*3+2);
   double ax=vertexData.get((i+1)*3)-x,ay=vertexData.get((i+1)*3+1)-y,az=vertexData.get((i+1)*3+2)-z;
   double bx=vertexData.get((i+2)*3)-x,by=vertexData.get((i+2)*3+1)-y,bz=vertexData.get((i+2)*3+2)-z;
   double dot=(ay*bz-az*by)*(cx-x)+(az*bx-ax*bz)*(cy-y)+(ax*by-ay*bx)*(cz-z);
   if(dot>0)frontVertices+=4;
  }
 }
 public static void glCallList(int id){if(frontSizes.getOrDefault(id,0)>0)drawnFrontLists.add(new int[]{frame,pass,id});}
 public static void glEnable(int a){}public static void glDisable(int a){}
 public static void glBindTexture(int a,int b){}public static void glTexParameteri(int a,int b,int c){}
 public static void glVertexPointer(int a,int b,FloatBuffer c){vertexData=c.duplicate();vertexData.clear();}public static void glTexCoordPointer(int a,int b,FloatBuffer c){}public static void glColorPointer(int a,int b,FloatBuffer c){}
 public static void glEnableClientState(int a){}public static void glDisableClientState(int a){}
 public static void glInitNames(){}public static void glPushName(int a){}public static void glPopName(){}
 public static void glBlendFunc(int a,int b){}public static void glColor4f(float a,float b,float c,float d){}
}
