// Runs the DOCUMENTED TMRP RECREATION unmodified, not the lost Notch build.
import java.lang.reflect.*;
import java.nio.file.*;
import java.util.*;
public class InspectRecreation {
 public static void main(String[] args) throws Exception {
  long t=System.nanoTime();
  Class<?> c=Class.forName("com.mojang.rubydung.level.Level");
  Object l=c.getConstructor(int.class,int.class,int.class).newInstance(256,256,64);
  Field f=c.getDeclaredField("blocks");f.setAccessible(true);byte[] b=(byte[])f.get(l);
  Field s=c.getDeclaredField("seed");s.setAccessible(true);
  int[] layers=new int[64];int sum=0;
  for(int y=0;y<64;y++)for(int z=0;z<256;z++)for(int x=0;x<256;x++)if(b[(y*256+z)*256+x]!=0){sum++;layers[y]++;}
  System.out.println("{\"artifact\":\"TMRP pc-131655 recreation; NOT original generator\",\"generated_seed\":"+s.getLong(l)+",\"generation_seconds\":"+((System.nanoTime()-t)/1e9)+",\"solid_blocks\":"+sum+",\"solid_by_y\":"+Arrays.toString(layers)+"}");
  Files.write(Path.of(args[0]),b);
 }
}
