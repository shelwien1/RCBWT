
#include <stdio.h>

typedef unsigned int   uint;
typedef unsigned char  byte;

uint flen( FILE* f ) {
  fseek( f, 0, SEEK_END );
  uint len = ftell(f);
  fseek( f, 0, SEEK_SET );
  return len;
}

enum {
  CUTOFF=8,
  STKSIZ=(8*sizeof(void*)-2)
};

template <class T>
  void SWAP( T& t1, T& t2 ) { T tmp=t1; t1=t2; t2=tmp; }

template <class CMP, class Element>
void sh_qsort( Element* array, uint lo, uint hi ) {
  // Note: the number of stack entries required is no more than
  //   1 + log2(num), so 30 is sufficient for any array
  uint lostk[STKSIZ];
  uint histk[STKSIZ];
  uint mid;                  /* points to middle of subarray */
  uint loguy;
  uint higuy;        /* traveling pointers for partition step */
  int  size;                /* size of the sub-array */
  int  stkptr;                 /* stack for saving sub-array to be processed */

  if( hi-lo+1<2 ) return;

  stkptr = 0;

recurse:

  size = hi - lo + 1;

  if( size<=CUTOFF ) {
    uint p, max;
    while( hi>lo ) {
      max = lo;
      for( p=lo+1; p<=hi; p++ ) if( CMP::c(array,p,max)>0 ) max = p;
      CMP::s(array, max, hi );
      hi--;
    }
  } else {
    mid = lo + (size>>1);
    if( CMP::c(array,lo, mid)>0 ) CMP::s(array,lo,mid);
    if( CMP::c(array,lo, hi)>0  ) CMP::s(array,lo,hi);
    if( CMP::c(array,mid, hi)>0 ) CMP::s(array,mid,hi);
    loguy = lo; higuy = hi;
    while(1) {
      if( mid>loguy )  do loguy++; while( loguy<mid && CMP::c(array,loguy,mid)<=0 );
      if( mid<=loguy ) do loguy++; while( loguy<=hi && CMP::c(array,loguy,mid)<=0 );
      do higuy--; while( higuy>mid && CMP::c(array,higuy,mid)>0 );
      if( higuy<loguy ) break;
      CMP::s(array,loguy,higuy);
      if( mid==higuy ) mid = loguy;
    }

    higuy++;
    if( mid<higuy )  do higuy--; while( higuy>mid && CMP::c(array,higuy,mid)==0 );
    if( mid>=higuy ) do higuy--; while( higuy>lo  && CMP::c(array,higuy,mid)==0 );

    if( higuy-lo >= hi-loguy ) {
      if( lo < higuy ) { lostk[stkptr]=lo; histk[stkptr]=higuy; ++stkptr; }
      if( loguy < hi ) { lo = loguy; goto recurse; }
    } else {
      if( loguy < hi ) { lostk[stkptr]=loguy; histk[stkptr]=hi; ++stkptr; }
      if( lo < higuy ) { hi = higuy; goto recurse; }
    }
  }

  if( --stkptr>=0 ) { lo = lostk[stkptr]; hi = histk[stkptr]; goto recurse; }
}

/*
struct TEST {
  // Returns neg if 1<2, 0 if 1=2, pos if 1>2.
  static int  c( pdirdata* restrict A, int x, int y ) {
    pdirdata a = A[x];
    pdirdata b = A[y];
//    return (a==b)?0:(a<b)?-1:1;
    return wcscmp( a->name, b->name );
  }
  static void s( pdirdata* restrict A, int x, int y ) {
    SWAP( A[x], A[y] );
  }
};
*/


byte* f_buf; uint f_len;
uint* f_idx;

struct TEST {
  // Returns neg if 1<2, 0 if 1=2, pos if 1>2.
  static int  c( uint* A, int x, int y ) {
//    if( x==y ) return 0;
    uint px = A[x];
    uint py = A[y];
    int r;
    for(; (r=1,px<f_len) && ((r=f_buf[px]-f_buf[py])==0); px=(++px>=f_len)?0:px,py=(++py>=f_len)?0:py );
    return r<0 ? -1 : 1;
  }
  static void s( uint* A, int x, int y ) {
    SWAP( A[x], A[y] );
  }
};

int main( int argc, char** argv ) {

  int i;

  if( argc<3 ) return 1;
  FILE* f = fopen( argv[1], "rb" );
  if( f==0 ) return 2;
  FILE* g = fopen( argv[2], "wb" );
  if( g==0 ) return 3;

  f_len = flen(f);
  printf( "filelength = %i\n", f_len);

  f_idx = new uint[f_len];
  if( f_idx==0 ) return 4;
  f_buf = new byte[f_len];
  if( f_buf==0 ) return 5;

  fread( f_buf, 1,f_len, f );
  fclose(f);

  printf( "data loading complete.\n" );

  for( i=0; i<f_len; i++ ) f_idx[i]=i;

  printf( "index init complete.\n" );

  sh_qsort<TEST>( f_idx, 0, f_len-1 );

  printf( "sort complete.\n" );

  int idx=0;
  for( i=0; i<f_len; i++ ) {
    int j = f_idx[i];
    if( j==0 ) idx=i, j=f_len-1; else j--;
    putc( f_buf[j], g );
  }
  fwrite( &idx, 1,sizeof(idx), g );
  fclose( g );

  printf( "BWT storing complete. index = %i\n", idx );

  return 0;
}
