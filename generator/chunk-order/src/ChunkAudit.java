import java.lang.reflect.*;import java.util.*;
import com.mojang.rubydung.level.*;
import org.lwjgl.opengl.GL11;
public class ChunkAudit {
 public static void main(String[] args)throws Exception{
  if(new java.io.File("level.dat").exists())throw new IllegalStateException("Run from a directory without level.dat: this experiment requires the original generated flat world.");
  double cx=169.993759155,cy=args.length>0?Double.parseDouble(args[0]):74,cz=116.979660034;double aspect=16.0/9;
  GL11.camera(cx,cy,cz,aspect);
  Level level=new Level(256,256,64);LevelRenderer renderer=new LevelRenderer(level);
  Field cs=LevelRenderer.class.getDeclaredField("chunks");cs.setAccessible(true);Chunk[] chunks=(Chunk[])cs.get(renderer);
  Field ls=Chunk.class.getDeclaredField("lists");ls.setAccessible(true);
  Map<Integer,Integer> index=new HashMap<>();for(int i=0;i<chunks.length;i++){int base=ls.getInt(chunks[i]);index.put(base,i);index.put(base+1,i);}
  List<Integer> visible=new ArrayList<>();Frustum f=Frustum.getFrustum();for(int i=0;i<chunks.length;i++)if(f.cubeInFrustum(chunks[i].aabb))visible.add(i);
  System.out.println("{\"type\":\"setup\",\"camera\":["+cx+","+cy+","+cz+"],\"aspect\":"+aspect+",\"visible_chunk_count\":"+visible.size()+",\"visible_array_order\":"+visible+"}");
  Field rf=Class.forName("java.lang.Math$RandomNumberGeneratorHolder").getDeclaredField("randomNumberGenerator");rf.setAccessible(true);Random global=(Random)rf.get(null);global.setSeed(12345);Random referenceRng=new Random(12345);
  for(int frame=0;frame<120;frame++)for(int pass=0;pass<2;pass++){GL11.frame=frame;GL11.pass=pass;renderer.render(null,pass);}
  System.out.println("{\"type\":\"rng_control\",\"next_global\":"+global.nextDouble()+",\"expected_next\":"+referenceRng.nextDouble()+"}");
  for(int[] r:GL11.compiled){int i=index.get(r[2]),layer=(r[2]-ls.getInt(chunks[i]));System.out.println("{\"type\":\"compile\",\"frame\":"+r[0]+",\"render_pass\":"+r[1]+",\"chunk_index\":"+i+",\"chunk_xyz\":["+((i/16)%16)+","+(i/256)+","+(i%16)+"],\"list_layer\":"+layer+",\"vertices\":"+r[3]+",\"front_facing_vertices\":"+r[4]+"}");}
  // Actual original Frustum method, sampled over potential horizontal origins.
  // Compare translated chunk sequences, not only known absolute chunk IDs.
  int n=24;String reference=signature(cx,cy,cz,aspect,chunks,n,true);List<String> matches=new ArrayList<>();
  for(int x=2;x<256;x+=4)for(int z=2;z<256;z+=4){if(signature(x,cy,z,aspect,chunks,n,true).equals(reference))matches.add("["+x+","+z+"]");}
  System.out.println("{\"type\":\"translated_order_grid_test\",\"prefix_chunks\":"+n+",\"reference\":\""+reference+"\",\"grid_spacing\":4,\"tested_positions\":4096,\"matching_count\":"+matches.size()+",\"matching_positions\":"+matches+"}");
 }
 static String signature(double x,double y,double z,double aspect,Chunk[] chunks,int n,boolean relative){
  GL11.camera(x,y,z,aspect);Frustum f=Frustum.getFrustum();StringBuilder b=new StringBuilder();int ox=0,oy=0,oz=0,count=0;
  for(int i=0;i<chunks.length&&count<n;i++)if(f.cubeInFrustum(chunks[i].aabb)){
   int xx=(i/16)%16,yy=i/256,zz=i%16;
   if(count==0){ox=xx;oy=yy;oz=zz;}
   b.append(xx-ox).append(',').append(yy-oy).append(',').append(zz-oz).append(';');count++;
  }return b.toString();
 }
}
