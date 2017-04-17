__kernel void transpose_naif (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [x * DIM + y] = in [y * DIM + x];
}



__kernel void transpose (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile [TILEX][TILEY+1];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile [xloc][yloc] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  out [(x - xloc + yloc) * DIM + y - yloc + xloc] = tile [yloc][xloc];
}

#define couleur 0xFFFF00FF // Yellow

/*
static int max(int a, int b)
{
  return a>b ? a : b;
}

static int min(int a, int b)
{
  return a<b ? a : b;
}*/

__kernel void vie_naif(__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  int nbAlive=0;
  for(int i = max(x-1, 0); i <= min(x+1, DIM-1); i++){
    for(int j = max(y-1, 0); j <= min(y+1, DIM-1); j++){
      if((i != x || j != y) && in[j * DIM + i] != 0){
        nbAlive++;
      }
    }
  }
  int isAlive = in [y * DIM + x] != 0;
  if(!isAlive && nbAlive==3){
    out [y * DIM + x] = couleur;
  }
  else if(isAlive && (nbAlive==2 || nbAlive==3)){
    out [y * DIM + x] = couleur;
  }
  else{
    out [y * DIM + x] = 0x00;
  }
}

__kernel void vie_opt(__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  int xloc = get_local_id(0);
  int yloc = get_local_id(1);

  __local int tuiles[TILEX+2][TILEY+2];

  tuiles[xloc+1][yloc+1]=in [y * DIM + x];

  if(xloc == 0 && x == 0){
    tuiles[0][yloc+1]=0x00;
  }
  else if(xloc==0){
    tuiles[0][yloc+1]=in [y * DIM + (x-1)];
  }

  if(yloc == 0 && y == 0){
    tuiles[xloc+1][0]=0x00;
  }
  else if(yloc == 0){
    tuiles[xloc+1][0]=in [(y-1) * DIM + x];
  }

  if(xloc == TILEX-1 && x == DIM-1){
    tuiles[xloc+2][yloc+1]=0x00;
  }
  else if(xloc == TILEX-1){
    tuiles[xloc+2][yloc+1]=in [y * DIM + (x+1)];
  }

  if(yloc == TILEY-1 && y == DIM-1){
    tuiles[xloc+1][yloc+2]=0x00;
  }
  else if(yloc == TILEY-1){
    tuiles[xloc+1][yloc+2]=in[(y+1) * DIM + x];
  }

  if(xloc == 0 && yloc == 0 && (x == 0 || y ==0)){
    tuiles[0][0]=0x00;
  }
  else if(xloc == 0 && yloc == 0){
    tuiles[0][0]=in[(y-1) * DIM + (x-1)];
  }

  if(xloc == 0 && yloc == TILEY-1 && (x == 0 || y == DIM-1)){
    tuiles[0][yloc+2]=0x00;
  }
  else if(xloc == 0 && yloc == TILEY-1){
    tuiles[0][yloc+2]=in[(y+1) * DIM + (x-1)];
  }

  if(xloc == TILEX-1 && yloc == 0 && (x == DIM-1) || y == 0){
    tuiles[xloc+2][0]=0x00;
  }
  else if(xloc == TILEX-1 && yloc ==0){
    tuiles[xloc+2][0]=in[(y-1) * DIM + (x+1)];
  }

  if(xloc == TILEX-1 && yloc == TILEY-1 && (x == DIM-1 || y == DIM-1)){
    tuiles[xloc+2][yloc+2]=0x00;
  }
  else if(xloc == TILEX-1 && yloc == TILEY-1){
    tuiles[xloc+2][yloc+2]=in[(y+1) * DIM + (x+1)];
  }


barrier(CLK_LOCAL_MEM_FENCE);

  int nbAlive=0;
  for(int i = xloc; i <= xloc+2; i++){
    for(int j = yloc; j <= yloc+2; j++){
      if((i != xloc+1 || j != yloc+1) && tuiles[i][j] != 0){
        nbAlive++;
      }
    }
  }
  int isAlive = tuiles [xloc+1][yloc+1] != 0;
  if(!isAlive && nbAlive==3){
    out [y * DIM + x] = couleur;
  }
  else if(isAlive && (nbAlive==2 || nbAlive==3)){
    out [y * DIM + x] = couleur;
  }
  else{
    out [y * DIM + x] = 0x00;
  }
}



// NE PAS MODIFIER
static float4 color_scatter (unsigned c)
{
  uchar4 ci;

  ci.s0123 = (*((uchar4 *) &c)).s3210;
  return convert_float4 (ci) / (float4) 255;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c;

  c = cur [y * DIM + x];

  write_imagef (tex, pos, color_scatter (c));
}
