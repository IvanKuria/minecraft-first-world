import java.lang.reflect.*;
import java.util.*;
import com.mojang.rubydung.Player;
import com.mojang.rubydung.level.Level;
import org.lwjgl.input.Keyboard;

/** Executes original Player ticks for competing reset/input hypotheses. */
public final class RespawnMotion {
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        Field rf=Class.forName("java.lang.Math$RandomNumberGeneratorHolder")
            .getDeclaredField("randomNumberGenerator");
        rf.setAccessible(true);
        Random global=(Random)rf.get(null);
        Level level=new Level(256,256,64);
        Method setPos=Player.class.getDeclaredMethod("setPos",float.class,float.class,float.class);
        setPos.setAccessible(true);
        for(String scenario:new String[]{"grounded_still","grounded_forward","grounded_jump","falling_terminal"}) {
            for(int held=1;held<=4;held++) {
                Keyboard.reset=Keyboard.jump=Keyboard.forward=false;
                global.setSeed(123456789L);
                Player p=new Player(level);
                setPos.invoke(p,128f,43.9f,128f);
                p.tick(); // settle on known y43 slab: eye y44.62, onGround true.
                if(scenario.equals("falling_terminal")) {p.yd=-.245f;p.onGround=false;}
                p.yRot=32f;p.xRot=15f;
                System.out.printf("{\"scenario\":\"%s\",\"reset_held_ticks\":%d,\"tick\":0,\"x\":%.9f,\"y\":%.9f,\"z\":%.9f,\"yd\":%.9f,\"on_ground\":%s}%n",
                    scenario,held,p.x,p.y,p.z,p.yd,p.onGround);
                for(int tick=1;tick<=60;tick++) {
                    Keyboard.reset=tick<=held;
                    Keyboard.jump=scenario.equals("grounded_jump");
                    Keyboard.forward=scenario.equals("grounded_forward");
                    p.tick();
                    System.out.printf("{\"scenario\":\"%s\",\"reset_held_ticks\":%d,\"tick\":%d,\"x\":%.9f,\"y\":%.9f,\"z\":%.9f,\"previous_y\":%.9f,\"xd\":%.9f,\"yd\":%.9f,\"zd\":%.9f,\"on_ground\":%s}%n",
                        scenario,held,tick,p.x,p.y,p.z,p.yo,p.xd,p.yd,p.zd,p.onGround);
                }
            }
        }
    }
}
