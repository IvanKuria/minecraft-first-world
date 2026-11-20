package org.lwjgl;
import java.nio.*;
public class BufferUtils {
 public static FloatBuffer createFloatBuffer(int n){return ByteBuffer.allocateDirect(n*4).order(ByteOrder.nativeOrder()).asFloatBuffer();}
 public static IntBuffer createIntBuffer(int n){return ByteBuffer.allocateDirect(n*4).order(ByteOrder.nativeOrder()).asIntBuffer();}
 public static ByteBuffer createByteBuffer(int n){return ByteBuffer.allocateDirect(n).order(ByteOrder.nativeOrder());}
}
