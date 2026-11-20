import java.lang.reflect.*;
import java.util.*;
import com.mojang.rubydung.Player;
import com.mojang.rubydung.level.*;
import org.lwjgl.opengl.GL11;

public class DynamicAudit {
 public static void main(String[] args) throws Exception {
  if(new java.io.File("level.dat").exists())throw new IllegalStateException("No level.dat allowed in calibration working directory.");
  double renderFps=args.length>0?Double.parseDouble(args[0]):72;
  int initialTicks=args.length>1?Integer.parseInt(args[1]):60;
  Level level=new Level(256,256,64);
  // Explicit artificial control world: all cells solid, top surface exactly y=64.
  Field bs=Level.class.getDeclaredField("blocks");bs.setAccessible(true);Arrays.fill((byte[])bs.get(level),(byte)1);
  level.calcLightDepths(0,0,256,256);
  Field rf=Class.forName("java.lang.Math$RandomNumberGeneratorHolder").getDeclaredField("randomNumberGenerator");rf.setAccessible(true);Random global=(Random)rf.get(null);global.setSeed(123456789L);
  Player p=new Player(level);Random ref=new Random(123456789L);ref.nextDouble();ref.nextDouble();
  for(int i=0;i<initialTicks;i++)p.tick();
  LevelRenderer renderer=new LevelRenderer(level);
  Field cf=LevelRenderer.class.getDeclaredField("chunks");cf.setAccessible(true);Chunk[] chunks=(Chunk[])cf.get(renderer);
  Field lf=Chunk.class.getDeclaredField("lists");lf.setAccessible(true);
  Map<Integer,Integer> ids=new HashMap<>();for(int i=0;i<chunks.length;i++){int id=lf.getInt(chunks[i]);ids.put(id,i);ids.put(id+1,i);}
  int simulatedTicks=0;
  for(int frame=0;frame<Math.ceil(renderFps*3);frame++){
   double t=frame/renderFps,ticks=t*60;int whole=(int)Math.floor(ticks);float a=(float)(ticks-whole);
   while(simulatedTicks<whole){p.tick();simulatedTicks++;}
   // Original camera's 0.3-unit setback at pitch=yaw=0 adds +0.3 to camera z.
   double x=p.xo+(p.x-p.xo)*a,y=p.yo+(p.y-p.yo)*a,z=p.zo+(p.z-p.zo)*a+.3f;
   GL11.camera(x,y,z,16.0/9);
   int previous=GL11.compiled.size(),prevDraw=GL11.drawnFrontLists.size();
   for(int pass=0;pass<2;pass++){GL11.frame=frame;GL11.pass=pass;renderer.render(p,pass);}
   List<Integer> compiled=new ArrayList<>(),frontDrawn=new ArrayList<>();
   for(int j=previous;j<GL11.compiled.size();j++){int idx=ids.get(GL11.compiled.get(j)[2]);if(!compiled.contains(idx))compiled.add(idx);}
   for(int j=prevDraw;j<GL11.drawnFrontLists.size();j++){int idx=ids.get(GL11.drawnFrontLists.get(j)[2]);if(!frontDrawn.contains(idx))frontDrawn.add(idx);}
   System.out.printf(Locale.ROOT,"{\"frame\":%d,\"seconds\":%.9f,\"render_fps\":%.3f,\"initial_ticks\":%d,\"camera\":[%.9f,%.9f,%.9f],\"new_chunks\":%s,\"drawn_front_facing_chunks\":%s}%n",frame,t,renderFps,initialTicks,x,y,z,compiled,frontDrawn);
  }
  System.err.println("RNG after constructor + ticks + rendering: "+global.nextDouble()+"; reference: "+ref.nextDouble());
 }
}
