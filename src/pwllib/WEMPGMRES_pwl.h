#ifndef WEMPGMRES_pwl
#define WEMPGMRES_pwl
/*****************
 *  WEMPGMRES_pwl.h  *
 *****************/


/*==============================================================*
 *  WEMPGMRES_pwl(A,b,x,epsi,W,F,p,M)                               *
 *	                                                        *
 *  GMRES-Verfahren zur Loesung des linearen Gleichungssystems  *
 *	                                                        *
 *		 A2'*x = b  bzw. A2*x = b.                      *
 *	                                                        *
 *  Vorkonditionierung per Diagonalskalierung.                  *
 *	                                                        *
 *  Parameter :                                                 *
 *		A    : Matrix im sparse2-Format                 *
 *		b    : rechte Seite                             *
 *		x    : Startwert und Endwert                    *
 *		epsi : Genauigkeit                              *
 *		W    : Liste der Wavelets                       *
 *		F    : Elementliste der Einskalenbasis          *
 *		p    : Anzahl der Paramtergebiete               *
 *		M    : Zahl der Level                           *
 *==============================================================*/


unsigned int WEMPGMRES_pwl1(sparse2 *A, double *b, double *x, double epsi, wavelet_pwl * W, unsigned int **F, unsigned int p, unsigned int M);


unsigned int WEMPGMRES2_pwl(sparse2 *A, double *b, double *x, double epsi, wavelet_pwl * W, unsigned int **F, unsigned int p, unsigned int M);


/*==============================================================*
 *  WEMPGMRES_pwl(A,B,rhs,x,epsi,W,F,p,M)                           *
 *	                                                        *
 *  GMRES-Verfahren zur Loesung des linearen Gleichungssystems  *
 *	                                                        *
 *	    (B1*G^(-1)*A2'-B2*G^(-1)*A1)*x = rhs.               *
 *	                                                        *
 *  Vorkonditionierung per Wavelet-Preconditioner.              *
 *	                                                        *
 *  Parameter :                                                 *
 *		A, B : Matrizen im sparse2-Format               *
 *		rhs  : rechte Seite                             *
 *		x    : Startwert und Endwert                    *
 *		epsi : Genauigkeit                              *
 *		W    : Liste der Wavelets                       *
 *		F    : Elementliste der Einskalenbasis          *
 *		p    : Anzahl der Paramtergebiete               *
 *		M    : Zahl der Level                           *
 *==============================================================*/

unsigned int WEMPGMRES_pwl3(sparse2 *A, sparse2 *B, double *rhs, double *x, double epsi, wavelet_pwl * W, unsigned int **F, unsigned int p, unsigned int M);
#endif
