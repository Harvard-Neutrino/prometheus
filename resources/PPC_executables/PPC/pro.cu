#ifdef XCPU
#define __device__
#define __global__

#define rsqrtf 1/sqrtf
#define __float2int_rn (int)lroundf
#define __float2int_ru (int)ceilf
#define __float2int_rd (int)floorf

struct int2{
  int x, y;
};

struct uint4{
  unsigned int x, y, z, w;
};

float int_as_float(unsigned int x){
  union{
    unsigned int i;
    float f;
  };
  i=x; return f;
}

unsigned int atomicAdd(unsigned int * i, unsigned int j){
  unsigned int k=*i; *i+=j;
  return k;
}

struct ThreadIdx{
  int x;
} threadIdx;

struct BlockDim{
  int x;
} blockDim;

unsigned int seed=0;

#if defined(__APPLE_CC__) || defined(__FreeBSD__)
void sincosf(float x, float * s, float * c){ *s = sin(x); *c = cos(x); }
#endif
#endif

__device__ float xrnd(uint4 & s){
  unsigned int tmp;
  do{
    unsigned long long sda;
#ifdef XCPU
    sda = s.z * (unsigned long long) s.x;
#else
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(sda) : "r"(s.x), "r"(s.z));
#endif
    sda += s.y; s.x = sda; s.y = sda >> 32; tmp = s.x >> 9;
  } while(tmp==0);
  return int_as_float(tmp|0x3f800000)-1.0f;
}

#ifdef LONG
__device__ float mrnd(float k, uint4 & s){  // gamma distribution
  float x;
  if(k<1){  // Weibull algorithm
    float c=1/k;
    float d=(1-k)*powf(k, k/(1-k));
    float z, e;
    do{
      z=-logf(xrnd(s));
      e=-logf(xrnd(s));
      x=powf(z, c);
    } while(z+e<d+x);
  }
  else{  // Cheng's algorithm
    float b=k-logf(4.0f);
    float l=sqrtf(2*k-1);
    float c=1+logf(4.5f);
    float u, v, y, z, r;
    do{
      u=xrnd(s); v=xrnd(s);
      y=logf(v/(1-v))/l;
      x=k*expf(y);
      z=u*v*v;
      r=b+(k+l)*y-x;
    } while(r<4.5f*z-c && r<logf(z));
  }
  return x;
}
#endif

__device__ void swap(float & x, float & y){
  float a=x; x=y; y=a;
}

__device__ void rotate(float & cs, float & si, float3 & n, uint4 & s){
  float3 p1, p2;
  int i=0;
  {
    float3 r;
    r.x=n.x*n.x, r.y=n.y*n.y, r.z=n.z*n.z;
    if(r.y>r.z){
      if(r.y>r.x) i=(swap(n.x,n.y),swap(r.x,r.y),1);
    }
    else{
      if(r.z>r.x) i=(swap(n.x,n.z),swap(r.x,r.z),2);
    }

    r.y=rsqrtf(r.x+r.y); p1.x=-n.y*r.y; p1.y=n.x*r.y; p1.z=0;
    r.z=rsqrtf(r.x+r.z); p2.x=-n.z*r.z; p2.y=0; p2.z=n.x*r.z;
  }

  {
    float4 q1;

    q1.x=p1.x-p2.x; q1.y=p1.y-p2.y; q1.z=p1.z-p2.z;
    p2.x+=p1.x; p2.y+=p1.y; p2.z+=p1.z;

    q1.w=rsqrtf(q1.x*q1.x+q1.y*q1.y+q1.z*q1.z);
    p1.x=q1.x*q1.w; p1.y=q1.y*q1.w; p1.z=q1.z*q1.w;

    q1.w=rsqrtf(p2.x*p2.x+p2.y*p2.y+p2.z*p2.z);
    p2.x*=q1.w; p2.y*=q1.w; p2.z*=q1.w;
  }

  {
    float2 p;
    float xi=2*FPI*xrnd(s);
    sincosf(xi, &p.y, &p.x);

    n.x=cs*n.x+si*(p.x*p1.x+p.y*p2.x);
    n.y=cs*n.y+si*(p.x*p1.y+p.y*p2.y);
    n.z=cs*n.z+si*(p.x*p1.z+p.y*p2.z);

    float r=rsqrtf(n.x*n.x+n.y*n.y+n.z*n.z);
    n.x*=r; n.y*=r; n.z*=r;
    if(i==1) swap(n.x,n.y); else if(i==2) swap(n.x,n.z);
  }
}

#ifdef TILT
#ifndef XCPU
__device__ int __float2int_rd(float x);
__host__ int __float2int_rd(float x){ return (int)floorf(x); }
__host__
#endif

__device__ float zshift(dats & d, float4 & r){
  if(d.lnum==0) return 0;
  float z=(r.z-d.lmin)*d.lrdz;
  int k=min(max(__float2int_rd(z), 0), d.lpts-2);
  int l=k+1;

  float nr=d.lnx*r.x+d.lny*r.y-d.r0;
  for(int j=1; j<LMAX; j++) if(nr<d.lr[j] || j==d.lnum-1){
    int i=j-1;
    return ( (d.lp[j][l]*(z-k)+d.lp[j][k]*(l-z))*(nr-d.lr[i]) +
	     (d.lp[i][l]*(z-k)+d.lp[i][k]*(l-z))*(d.lr[j]-nr) )/(d.lr[j]-d.lr[i]);
  }
  return 0;
}
#endif

__device__ void ctr(dats & d, float2 & r, float2 & p){
#ifdef ROMB
  p.x=d.cb[0][0]*r.x+d.cb[1][0]*r.y;
  p.y=d.cb[0][1]*r.x+d.cb[1][1]*r.y;
#else
  p=r;
#endif
}

#ifdef XCPU
DOM * oms;
#else
__constant__ DOM oms[MAXGEO];

__device__ inline unsigned int smid(){
  unsigned int r;
  asm("mov.u32 %0, %smid;" : "=r"(r));
  return r;
}
#endif

#if defined(USMA) && defined(RAND)
#define XINC i=atomicAdd(&eidx, e.gridDim)
#define XIDX e.gridDim*blockDim.x+e.blockIdx
#else
#define XINC i+=eidx
#define XIDX e.gridDim*blockDim.x
#endif

#ifdef HOLE
#define IFH(x,y) x
#else
#define IFH(x,y) y
#endif

__global__ void propagate(dats * ed, unsigned int num){
  uint4 s;
  unsigned int niw=0;
#ifdef XCPU
  float3 n;
  float4 r;
  dats & e = * ed;
  static unsigned int eidx;
  if(threadIdx.x==0) eidx = XIDX;
#else
  float3 n={0,0,0};
  float4 r={0,0,0,0};
  __shared__ dats e;
  unsigned int & eidx = e.hidx;

  if(num==0){
    ed->hidx=0;
    ed->tn=-1U;
    ed->tx=0;
    ed->ab=0;
    ed->mp=0;
    __threadfence();
    return;
  }

  if(threadIdx.x==0){
    e=*ed; e.tn=clock();
    e.blockIdx=smid()==e.blockIdx?-1:(int)atomicAdd(&ed->mp, 1);
    eidx=XIDX;
  }
  __syncthreads();

  if(e.blockIdx==-1) return;
#endif

  ices * w;
  const unsigned int idx=threadIdx.x*e.gridDim+e.blockIdx;

  {
#ifndef XCPU
    const unsigned int & seed = idx;
#endif
    s.w=seed%e.rsize;
    s.x=e.z->rs[s.w];
    s.y=e.z->rs[s.w] >> 32;
    s.z=e.z->rm[s.w];
  }

  int old;
  float TOT=0, IFH(SCA,sca);

#ifdef TALL
  for(unsigned int i=idx; i<num; i+=e.gridDim*blockDim.x){
#else
  for(unsigned int i=idx; i<num; TOT==0 && (XINC)){
    int om=-1;
    if(TOT==0){ // initialize photon
#endif

      unsigned int j=min(__float2int_rd(WNUM*xrnd(s)), WNUM-1);
      w=&e.z->w[j];
      if(e.type>0){
	r.x=e.r[0];
	r.y=e.r[1];
	r.z=e.r[2];
	r.w=0;

	float xi=xrnd(s);
	if(e.fldr<0) xi*=2*FPI;
	else{
	  int r=__float2int_rd(e.fldr/360)+1;
	  int s=__float2int_rd(xi*r);
	  xi=(e.fldr+s*360/r)*fcv;
	}
	sincosf(xi, &n.y, &n.x);

	if(e.ka>0){
	  float FLZ, FLR;
	  sincosf(fcv*30.f, &FLZ, &FLR);
	  FLZ*=OMR, FLR*=OMR;
	  r.x+=FLR*n.x;
	  r.y+=FLR*n.y;
	  r.z+=FLZ;
	  r.w+=OMR*w->ocm;
	}

	sincosf(e.up, &n.z, &xi);
	n.x*=xi; n.y*=xi;

	if(e.ka>0){
	  do{ xi=1+e.ka*logf(xrnd(s)); } while (xi<-1);
	  float si=sqrtf(1-xi*xi); rotate(xi, si, n, s);
	}
      }
      else{
	photon p=e.pz[i/OVER]; r=p.r; n=p.n;
	float & l=p.l;
#ifndef TALL
	niw=p.q;
#endif

	if(l<0){
	  float xi;
	  if(e.ka>0){
	    do{ xi=1+e.ka*logf(xrnd(s)); } while (xi<-1);
	    float si=sqrtf(1-xi*xi); rotate(xi, si, n, s);
	  }
	}
	else{
	  if(l>0) l*=xrnd(s);
#ifdef LONG
	  else if(p.b>0) l=p.b*mrnd(p.a, s);
#endif
	  if(l>0){
	    r.w+=e.ocv*l;
	    r.x+=n.x*l; r.y+=n.y*l; r.z+=n.z*l;
	  }

#ifdef ANGW
	  if(p.f<xrnd(s)){
	    const float a=0.39f, b=2.61f;
	    const float I=1-expf(-b*exp2f(a));
	    float cs=max(1-powf(-logf(1-xrnd(s)*I)/b, 1/a), -1.0f);
	    float si=sqrtf(1-cs*cs); rotate(cs, si, n, s);
	  }
#endif
	  rotate(w->coschr, w->sinchr, n, s);
	}
      }

#ifdef TALL
      pbuf f; f.r=r, f.n=n, f.q=j; e.bf[i]=f;
  }
#ifndef XCPU
  __threadfence_block();
#endif

  for(unsigned int i=idx; i<num; TOT==0 && (XINC)){
    int om=-1;
    if(TOT==0){ // initialize photon
      pbuf f=e.bf[i];
      r=f.r; n=f.n; w=&e.z->w[f.q];
      if(e.type<=0) niw=e.pz[i/OVER].q;
#endif
      om=e.fla;

      TOT=-logf(xrnd(s)), IFH(SCA,sca)=0;
    }

#ifdef HOLE
    if(SCA==0) SCA=-logf(xrnd(s)), old=om;
    float sca, tot;
#else
    if(sca==0){ // get distance for overburden
      float SCA=-logf(xrnd(s)); old=om;
      float tot;
#endif
      float z = r.z;
#ifdef TILT
      z -= zshift(e, r);
#endif

      float nr=1.f;
#ifdef ANIZ
      if(e.k>0){
	float n1= e.azx*n.x+e.azy*n.y;
	float n2=-e.azy*n.x+e.azx*n.y;
	float n3= n.z;

	float s1=n1*n1, l1=e.k1*e.k1;
	float s2=n2*n2, l2=e.k2*e.k2;
	float s3=n3*n3, l3=e.kz*e.kz;

	float B2=nr/l1+nr/l2+nr/l3;
	float nB=s1/l1+s2/l2+s3/l3;
	float An=s1*l1+s2*l2+s3*l3;

	nr=(B2-nB)*An/2;
	TOT/=nr;
      }
#endif

#ifdef HOLE
    {
#endif
      int i=__float2int_rn((z-e.hmin)*e.rdh);
      if(i<0) i=0; else if(i>=e.size) i=e.size-1;
      float h=e.hmin+i*e.dh; // middle of the layer
      h=n.z<0?h-e.hdh:h+e.hdh;

      float ais=(n.z*SCA-(h-z)*w->z[i].sca)*e.rdh;
      float aia=(n.z*TOT-(h-z)*w->z[i].abs)*e.rdh;

      int j=i;
      if(n.z<0) for(; j>0 && ais<0 && aia<0; h-=e.dh, ais+=w->z[j].sca, aia+=w->z[j].abs) --j;
      else for(; j<e.size-1 && ais>0 && aia>0; h+=e.dh, ais-=w->z[j].sca, aia-=w->z[j].abs) ++j;

      if(i==j || fabsf(n.z)<XXX) sca=SCA/w->z[j].sca, tot=TOT/w->z[j].abs;
      else sca=(ais*e.dh/w->z[j].sca+h-z)/n.z, tot=(aia*e.dh/w->z[j].abs+h-z)/n.z;

      // get overburden for distance
      if(tot<sca) sca=tot, IFH(tot,TOT)=0; else IFH(tot=,TOT=nr*)(tot-sca)*w->z[j].abs;
    }

    om=-1;
    float del=sca;
#ifdef HOLE
    float hi=sca, hf=0;
#endif
    { // sphere
#ifndef HOLE
      float & sca = del;
#endif
      float2 ri, rf, pi, pf;

      ri.x=r.x; rf.x=r.x+sca*n.x;
      ri.y=r.y; rf.y=r.y+sca*n.y;

      ctr(e, ri, pi); ctr(e, rf, pf);

      ri.x=min(pi.x, pf.x)-e.rx; rf.x=max(pi.x, pf.x)+e.rx;
      ri.y=min(pi.y, pf.y)-e.rx; rf.y=max(pi.y, pf.y)+e.rx;

      int2 xl, xh;

      xl.x=min(max(__float2int_rn((ri.x-e.cl[0])*e.crst[0]), 0), e.cn[0]);
      xh.x=max(min(__float2int_rn((rf.x-e.cl[0])*e.crst[0]), e.cn[0]-1), -1);

      xl.y=min(max(__float2int_rn((ri.y-e.cl[1])*e.crst[1]), 0), e.cn[1]);
      xh.y=max(min(__float2int_rn((rf.y-e.cl[1])*e.crst[1]), e.cn[1]-1), -1);

      for(int i=xl.x, j=xl.y; i<=xh.x && j<=xh.y; ++j<=xh.y?:(j=xl.y,i++)) for(unsigned char k=e.is[i][j]; k!=0x80; ){
	unsigned char m=e.ls[k];
	line & s = e.sc[m&0x7f];
	k=m&0x80?0x80:k+1;

	float b=0, c=0, dr;
	dr=s.x-r.x;
	b+=n.x*dr; c+=dr*dr;
	dr=s.y-r.y;
	b+=n.y*dr; c+=dr*dr;

	float np=1-n.z*n.z;
	float D=b*b-(c-s.r*s.r)*np;
	if(D>=0){
	  D=sqrtf(D);
	  float h1=b-D, h2=b+D;
	  if(h2>=0 && h1<=sca*np){
	    if(np>XXX){
	      h1/=np, h2/=np;
	      if(h1<0) h1=0; if(h2>sca) h2=sca;
	    }
	    else h1=0, h2=sca;
	    h1=r.z+n.z*h1, h2=r.z+n.z*h2;
	    float zl, zh;
	    if(n.z>0) zl=h1, zh=h2;
	    else zl=h2, zh=h1;

	    int omin=0, omax=s.max;
	    int n1=s.n-omin+min(omax+1, max(omin, __float2int_ru(omin-(zh-s.dl-s.h)*s.d)));
	    int n2=s.n-omin+max(omin-1, min(omax, __float2int_rd(omin-(zl-s.dh-s.h)*s.d)));

	    for(int l=n1; l<=n2; l++) if(l!=old){
#ifdef OFLA
	      if(l==e.fla) continue;
#endif
	      const DOM & dom=oms[l];
	      float b=0, c=0, dr;
	      dr=dom.r[0]-r.x;
	      b+=n.x*dr; c+=dr*dr;
	      dr=dom.r[1]-r.y;
	      b+=n.y*dr; c+=dr*dr;
	      dr=dom.r[2]-r.z;
	      b+=n.z*dr; c+=dr*dr;
	      float D=b*b-c+e.R2;
	      if(D>=0){
		float h=b-sqrtf(D)*e.zR;
		if(h>0 && h<=del) om=l, del=h;
	      }
	    }
	  }
#ifdef HOLE
	  if(e.hr>0){
	    float D=b*b-(c-e.hr2)*np;
	    if(D>0){
	      D=sqrtf(D);
	      float h1=b-D, h2=b+D;
	      if(h2>=0 && h1<=sca*np){
		if(np>XXX){
		  h1/=np, h2/=np;
		  if(h1<0) h1=0; if(h2>sca) h2=sca;
		}
		else h1=0, h2=sca;
		if(h1<hi && h2>sqrtf(XXX)*e.hr) hi=h1, hf=h2;
	      }
	    }
	  }
#endif
	}
      }
    }

#ifdef HOLE
    float fin=min(del, hi);
    bool hole=fin<sca;
    if(hole){
      { // get overburden for distance
	float xs=0, xa=0;
	int i=__float2int_rn((z-e.hmin)*e.rdh);
	if(i<0) i=0; else if(i>=e.size) i=e.size-1;

	float y = z + n.z*fin;
	int j=__float2int_rn((y-e.hmin)*e.rdh);
	if(j<0) j=0; else if(j>=e.size) j=e.size-1;

	if(i==j || fabsf(n.z)<XXX) xs=fin*w->z[i].sca, xa=fin*w->z[i].abs;
	else{
	  int k=j;
	  float h=e.hmin+i*e.dh, g=e.hmin+j*e.dh;
	  if(n.z<0){
	    h-=e.hdh, g+=e.hdh;
	    while(++k<i) xs-=w->z[k].sca, xa-=w->z[k].abs;
	  }
	  else{
	    h+=e.hdh, g-=e.hdh;
	    while(--k>i) xs+=w->z[k].sca, xa+=w->z[k].abs;
	  }
	  xs=((y-g)*w->z[j].sca+(h-z)*w->z[i].sca+e.dh*xs)/n.z;
	  xa=((y-g)*w->z[j].abs+(h-z)*w->z[i].abs+e.dh*xa)/n.z;
	}
	SCA-=xs, TOT-=xa;
      }
      TOT*=nr;

      if(hi<del){
	fin=min(hi+min(SCA/e.hs, TOT/e.ha), hf);
	if(fin<del) del=fin, om=-1;
	fin-=hi; SCA-=fin*e.hs, TOT-=fin*e.ha;
      }
    }
    else SCA=0, TOT=tot*nr;
#else
    sca-=del;
#endif

    { // advance
      r.x+=del*n.x;
      r.y+=del*n.y;
      r.z+=del*n.z;
      r.w+=del*w->ocm;
    }

#ifndef XCPU
    if(!isfinite(TOT) || !isfinite(IFH(SCA,sca))) ed->bmp[atomicAdd(&ed->ab, 1)%4]=smid(), TOT=0, om=-1;
#endif

    float xi=xrnd(s);
    if(om!=-1){
      bool flag=true;
      hit h; h.i=om; h.t=r.w; h.n=niw; h.z=w->wvl;

#ifdef ASENS
      float sum;
      {
	float & x = n.z;
	float y=1;
	sum=e.s[0];
	for(int i=1; i<ANUM; i++){ y*=x; sum+=e.s[i]*y; }
      }

      flag=e.mas*xi<sum;
#endif
      if(e.type>0){
	float dt=0, dr;
	const DOM & dom=oms[om];
	for(int i=0; i<3; i++, dt+=dr*dr) dr=dom.r[i]-e.r[i];
	if(h.t<(sqrtf(dt)-OMR)*w->ocm) flag=false;
      }

      if(flag){
	unsigned int j = atomicAdd(&ed->hidx, 1);
	if(j<e.hnum) e.hits[j]=h;
      }

      if(e.zR==1) TOT=0; else old=om;
    }
    else if(TOT<XXX) TOT=0;
#ifdef HOLE
    else if(SCA<XXX){
      SCA=0;
      float sf, g, g2, gr;
      if(hole){
	sf=e.SF, g=e.G, g2=e.G2, gr=e.GR;
      }
      else{
	sf=e.sf, g=e.g, g2=e.g2, gr=e.gr;
      }
#else
    else{
      float &sf=e.sf, &g=e.g, &g2=e.g2, &gr=e.gr;
#endif

      if(xi>sf){
	xi=(1-xi)/(1-sf);
	xi=2*xi-1;
	if(g!=0){
	  float ga=(1-g2)/(1+g*xi);
	  xi=(1+g2-ga*ga)/(2*g);
	}
      }
      else{
	xi/=sf;
	xi=2*powf(xi, gr)-1;
      }

      if(xi>1) xi=1; else if(xi<-1) xi=-1;

#ifdef ANIZ
      if(e.k>0 IFH(&& !hole,)){
	float n1=( e.azx*n.x+e.azy*n.y)*e.k1;
	float n2=(-e.azy*n.x+e.azx*n.y)*e.k2;
	float nx=n1*e.azx-n2*e.azy;
	float ny=n1*e.azy+n2*e.azx;
	float nz=n.z*e.kz;
	float r=rsqrtf(nx*nx+ny*ny+nz*nz);
	n.x=r*nx, n.y=r*ny, n.z=r*nz;
      }
#endif

      float si=sqrtf(1-xi*xi);
      rotate(xi, si, n, s);

#ifdef ANIZ
      if(e.k>0 IFH(&& !hole,)){
	float n1=( e.azx*n.x+e.azy*n.y)/e.k1;
	float n2=(-e.azy*n.x+e.azx*n.y)/e.k2;
	float nx=n1*e.azx-n2*e.azy;
	float ny=n1*e.azy+n2*e.azx;
	float nz=n.z/e.kz;
	float r=rsqrtf(nx*nx+ny*ny+nz*nz);
	n.x=r*nx, n.y=r*ny, n.z=r*nz;
      }
#endif
    }
  }

  {
    e.z->rs[s.w]=s.x | (unsigned long long) s.y << 32;
#ifndef XCPU
    __syncthreads();
    if(threadIdx.x==0){
      e.tx=clock();
      atomicMin(&ed->tn, e.tx-e.tn);
      atomicMax(&ed->tx, e.tx-e.tn);
    }
    __threadfence();
#endif
  }

}
