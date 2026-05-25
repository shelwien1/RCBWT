
#include <stdio.h>

typedef unsigned int   uint;
typedef unsigned char  byte;

template <class T, int N> void bzero( T (&p)[N] ) { int i; for( i=0; i<N; i++ ) p[i]=0; }

uint flen( FILE* f ) {
  fseek( f, 0, SEEK_END );
  uint len = ftell(f);
  fseek( f, 0, SEEK_SET );
  return len;
}

byte* f_buf; uint f_len; int f_lenl; uint f_lena;
uint* f_idx;

const int CNUM = 256;
uint freq[CNUM+1];

int main( int argc, char** argv ) {

  int i;

  if( argc<3 ) return 1;
  FILE* f = fopen( argv[1], "rb" );
  if( f==0 ) return 2;
  FILE* g = fopen( argv[2], "wb" );
  if( g==0 ) return 3;

  f_len = flen(f)-4;
  printf( "filelength = %i\n", f_len );

  f_idx = new uint[f_len];
  if( f_idx==0 ) return 4;
  f_buf = new byte[f_len+4];
  if( f_buf==0 ) return 5;

  fread( f_buf, 1,f_len+4, f );
  fclose(f);

  printf( "data loading complete.\n" );

  int idx = (int&)f_buf[f_len];
  printf( "index = %i\n", idx );

  bzero(freq);
  for( i=0; i<f_len; i++ ) freq[f_buf[i]+1]++;
  for( i=0; i<CNUM; i++ ) freq[i+1]+=freq[i];

  for( i=0; i<f_len; i++ ) {
    int c = f_buf[i];
//    f_idx[i] = freq[c]++;
    f_idx[freq[c]++]=i;
  }

  for( i=0; i<f_len; i++ ) {
    idx = f_idx[idx];
    putc( f_buf[idx], g );
  }

  printf( "unBWT storing complete.\n" );

  return 0;
}
