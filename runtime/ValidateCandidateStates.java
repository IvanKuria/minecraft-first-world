import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.Random;

/** Independent Java-runtime check of every retained scenario candidate. */
public final class ValidateCandidateStates {
    public static void main(String[] args) throws Exception {
        String input = new String(Files.readAllBytes(Paths.get(args[0])), StandardCharsets.UTF_8).trim();
        String[] states = input.substring(1, input.length()-1).split(",");
        double[][] absolute = {{195,201},{188,250},{222,225},{113,135}};
        double[][] fractional = {{.42,.50},{.82,.92},{.78,.85},{.79,.86}};
        int failures = 0;
        for (String text : states) {
            long internal = Long.parseLong(text.trim());
            Random random = new Random(internal ^ 0x5DEECE66DL);
            boolean valid = true;
            for (int i=0; i<4; i++) {
                double coordinate = (float)random.nextDouble() * 256;
                double phase = coordinate - Math.floor(coordinate);
                valid &= coordinate >= absolute[i][0]-1e-12 && coordinate <= absolute[i][1]+1e-12;
                valid &= phase >= fractional[i][0]-1e-12 && phase <= fractional[i][1]+1e-12;
            }
            if (!valid) failures++;
        }
        System.out.println("{\"checked\":"+states.length+",\"failed\":"+failures+",\"runtime\":\""+System.getProperty("java.version")+"\",\"historical_authentication\":false}");
        if (failures != 0) throw new AssertionError("Candidate constraints failed");
    }
}
