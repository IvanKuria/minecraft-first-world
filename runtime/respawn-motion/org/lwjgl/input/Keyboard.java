package org.lwjgl.input;
/** Deterministic input adapter; the preserved Player bytecode is unmodified. */
public final class Keyboard {
    public static boolean reset, jump, forward;
    public static boolean isKeyDown(int code) {
        return (code==19 && reset) || (code==57 && jump) || (code==17 && forward);
    }
}
