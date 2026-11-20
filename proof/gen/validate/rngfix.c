#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/jrand.h"
int main(int c,char**v){uint64_t s=strtoull(v[1],0,0)&JR_MASK;int n=c>2?atoi(v[2]):20;JRand r={s,0,0};
for(int i=0;i<n;i++){int a=jr_nextInt(&r,256),b=jr_nextInt(&r,64),d=jr_nextInt(&r,6),e=jr_nextInt(&r,2),f=jr_nextInt(&r,4194304),g=jr_nextInt(&r,1000);double x=jr_nextDouble(&r);float y=jr_nextFloat(&r);int64_t xb;int32_t yb;memcpy(&xb,&x,8);memcpy(&yb,&y,4);
printf("%d %d %d %d %d %d %lld %d\n",a,b,d,e,f,g,(long long)xb,yb);}printf("calls %llu\n",(unsigned long long)r.steps);}
