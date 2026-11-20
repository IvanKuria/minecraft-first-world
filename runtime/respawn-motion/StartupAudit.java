import java.lang.reflect.*;
import java.util.Arrays;
import java.util.Locale;
import com.mojang.rubydung.Player;
import com.mojang.rubydung.Timer;
import com.mojang.rubydung.level.Level;

/** Synthetic solid-y63 surface; actual Timer/Player classes, no sleep required. */
public final class StartupAudit {
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        Level level=new Level(256,256,64);
        Field blocks=Level.class.getDeclaredField("blocks");blocks.setAccessible(true);
        Arrays.fill((byte[])blocks.get(level),(byte)1);
        Player player=new Player(level);
        Timer timer=new Timer(60f);
        Field last=Timer.class.getDeclaredField("lastTime");last.setAccessible(true);
        last.setLong(timer,System.nanoTime()-2_000_000_000L);
        timer.advanceTime();
        System.out.printf("{\"kind\":\"timer\",\"simulated_initial_delay_at_least_seconds\":2,\"ticks\":%d,\"interpolation\":%.9f}%n",timer.ticks,timer.a);
        float initialX=player.x,initialZ=player.z;
        for(int tick=1;tick<=150;tick++) {
            player.tick();
            if(tick==1 || tick==60 || player.onGround) {
                System.out.printf("{\"kind\":\"player\",\"tick\":%d,\"y\":%.9f,\"yd\":%.9f,\"on_ground\":%s,\"x_change\":%.9f,\"z_change\":%.9f}%n",tick,player.y,player.yd,player.onGround,player.x-initialX,player.z-initialZ);
            }
            if(player.onGround)break;
        }
    }
}
