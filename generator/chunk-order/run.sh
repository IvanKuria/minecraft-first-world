#!/bin/sh
set -eu
cd "$(dirname "$0")"
JAVA_BIN=${JAVA_BIN:-/opt/homebrew/opt/openjdk/bin}
JAR=../../artifacts/rd-132211.mojang.jar
mkdir -p classes
"$JAVA_BIN/javac" -cp "$JAR" -d classes src/org/lwjgl/BufferUtils.java src/org/lwjgl/opengl/GL11.java src/org/lwjgl/input/Keyboard.java src/org/lwjgl/util/glu/GLU.java src/ChunkAudit.java src/DynamicAudit.java
"$JAVA_BIN/java" --add-opens java.base/java.lang=ALL-UNNAMED -Djava.awt.headless=true -cp "classes:$JAR" ChunkAudit > trace.jsonl 2> stderr.txt
python3 summarize.py
"$JAVA_BIN/java" --add-opens java.base/java.lang=ALL-UNNAMED -Djava.awt.headless=true -cp "classes:$JAR" ChunkAudit 65.62 > trace-grounded.jsonl 2> stderr-grounded.txt
"$JAVA_BIN/java" --add-opens java.base/java.lang=ALL-UNNAMED -Djava.awt.headless=true -cp "classes:$JAR" DynamicAudit 72 60 > dynamic-72fps.jsonl 2> dynamic-72fps.stderr.txt
"$JAVA_BIN/java" --add-opens java.base/java.lang=ALL-UNNAMED -Djava.awt.headless=true -cp "classes:$JAR" DynamicAudit 30 60 > dynamic-30fps.jsonl 2> dynamic-30fps.stderr.txt
