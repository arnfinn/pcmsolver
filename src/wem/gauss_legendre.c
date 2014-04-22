/* warning-disabler-start */

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wextra"
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#pragma warning push
#pragma warning disable "-Wall"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Weffc++"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdeprecated-register"
#pragma clang diagnostic ignored "-Wincompatible-pointer-types"
#pragma clang diagnostic ignored "-Wempty-body"
#endif

/* warning-disabler-end */

/**********************
 *  Gauss_Legendre.c  *
 **********************/


/*=======================================*
 *  Stuetzstellen Xi und Gewichte G der  *
 *  Gauss-Quadraturformeln auf [0,1].    *
 *=======================================*/


#include <stdio.h>
#include <stdlib.h>
#include "quadrature.h"
#include "gauss_legendre.h"


void Regel01_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 1 */
quadrature *Q;
{
    Q->nop = 1;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 5.0000000000000000000e-01;
    Q->w[0] = 1.0000000000000000000e+00;

    return;
}


void Regel02_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 3 */
quadrature *Q;
{
    Q->nop = 2;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 2.11324865405187117745e-01;
    Q->xi[1] = 7.88675134594812882255e-01;

    Q->w[0] = 5.00000000000000000000e-01;
    Q->w[1] = 5.00000000000000000000e-01;

    return;
}


void Regel03_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 5 */
quadrature *Q;
{
    Q->nop = 3;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 1.12701665379258311482e-01;
    Q->xi[1] = 5.00000000000000000000e-01;
    Q->xi[2] = 8.87298334620741688518e-01;

    Q->w[0] = 2.77777777777777777778e-01;
    Q->w[1] = 4.44444444444444444444e-01;
    Q->w[2] = 2.77777777777777777778e-01;

    return;
}


void Regel04_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 7 */
quadrature *Q;
{
    Q->nop = 4;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 6.94318442029737123880e-02;
    Q->xi[1] = 3.30009478207571867599e-01;
    Q->xi[2] = 6.69990521792428132401e-01;
    Q->xi[3] = 9.30568155797026287612e-01;

    Q->w[0] = 1.73927422568726928687e-01;
    Q->w[1] = 3.26072577431273071313e-01;
    Q->w[2] = 3.26072577431273071313e-01;
    Q->w[3] = 1.73927422568726928687e-01;

    return;
}


void Regel05_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 9 */
quadrature *Q;
{
    Q->nop = 5;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 4.69100770306680036012e-02;
    Q->xi[1] = 2.30765344947158454482e-01;
    Q->xi[2] = 5.00000000000000000000e-01;
    Q->xi[3] = 7.69234655052841545518e-01;
    Q->xi[4] = 9.53089922969331996399e-01;

    Q->w[0] = 1.18463442528094543757e-01;
    Q->w[1] = 2.39314335249683234021e-01;
    Q->w[2] = 2.84444444444444444444e-01;
    Q->w[3] = 2.39314335249683234021e-01;
    Q->w[4] = 1.18463442528094543757e-01;

    return;
}


void Regel06_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 11 */
quadrature *Q;
{
    Q->nop = 6;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 3.37652428984239860938e-02;
    Q->xi[1] = 1.69395306766867743169e-01;
    Q->xi[2] = 3.80690406958401545685e-01;
    Q->xi[3] = 6.19309593041598454315e-01;
    Q->xi[4] = 8.30604693233132256831e-01;
    Q->xi[5] = 9.66234757101576013906e-01;

    Q->w[0] = 8.56622461895851725201e-02;
    Q->w[1] = 1.80380786524069303785e-01;
    Q->w[2] = 2.33956967286345523695e-01;
    Q->w[3] = 2.33956967286345523695e-01;
    Q->w[4] = 1.80380786524069303785e-01;
    Q->w[5] = 8.56622461895851725201e-02;

    return;
}


void Regel07_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 13 */
quadrature *Q;
{
    Q->nop = 7;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 2.54460438286207377369e-02;
    Q->xi[1] = 1.29234407200302780068e-01;
    Q->xi[2] = 2.97077424311301416547e-01;
    Q->xi[3] = 5.00000000000000000000e-01;
    Q->xi[4] = 7.02922575688698583453e-01;
    Q->xi[5] = 8.70765592799697219932e-01;
    Q->xi[6] = 9.74553956171379262263e-01;

    Q->w[0] = 6.47424830844348466353e-02;
    Q->w[1] = 1.39852695744638333951e-01;
    Q->w[2] = 1.90915025252559472475e-01;
    Q->w[3] = 2.08979591836734693878e-01;
    Q->w[4] = 1.90915025252559472475e-01;
    Q->w[5] = 1.39852695744638333951e-01;
    Q->w[6] = 6.47424830844348466353e-02;

    return;
}


void Regel08_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 15 */
quadrature *Q;
{
    Q->nop = 8;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 1.98550717512318841582e-02;
    Q->xi[1] = 1.01666761293186630204e-01;
    Q->xi[2] = 2.37233795041835507091e-01;
    Q->xi[3] = 4.08282678752175097530e-01;
    Q->xi[4] = 5.91717321247824902470e-01;
    Q->xi[5] = 7.62766204958164492909e-01;
    Q->xi[6] = 8.98333238706813369796e-01;
    Q->xi[7] = 9.80144928248768115842e-01;

    Q->w[0] = 5.06142681451881295763e-02;
    Q->w[1] = 1.11190517226687235272e-01;
    Q->w[2] = 1.56853322938943643669e-01;
    Q->w[3] = 1.81341891689180991483e-01;
    Q->w[4] = 1.81341891689180991483e-01;
    Q->w[5] = 1.56853322938943643669e-01;
    Q->w[6] = 1.11190517226687235272e-01;
    Q->w[7] = 5.06142681451881295763e-02;

    return;
}


void Regel09_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 17 */
quadrature *Q;
{
    Q->nop = 9;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 1.59198802461869550822e-02;
    Q->xi[1] = 8.19844463366821028503e-02;
    Q->xi[2] = 1.93314283649704801346e-01;
    Q->xi[3] = 3.37873288298095535481e-01;
    Q->xi[4] = 5.00000000000000000000e-01;
    Q->xi[5] = 6.62126711701904464519e-01;
    Q->xi[6] = 8.06685716350295198654e-01;
    Q->xi[7] = 9.18015553663317897150e-01;
    Q->xi[8] = 9.84080119753813044918e-01;

    Q->w[0] = 4.06371941807872059860e-02;
    Q->w[1] = 9.03240803474287020292e-02;
    Q->w[2] = 1.30305348201467731159e-01;
    Q->w[3] = 1.56173538520001420034e-01;
    Q->w[4] = 1.65119677500629881582e-01;
    Q->w[5] = 1.56173538520001420034e-01;
    Q->w[6] = 1.30305348201467731159e-01;
    Q->w[7] = 9.03240803474287020292e-02;
    Q->w[8] = 4.06371941807872059860e-02;

    return;
}


void Regel10_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 19 */
quadrature *Q;
{
    Q->nop = 10;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 1.30467357414141399610e-02;
    Q->xi[1] = 6.74683166555077446340e-02;
    Q->xi[2] = 1.60295215850487796883e-01;
    Q->xi[3] = 2.83302302935376404600e-01;
    Q->xi[4] = 4.25562830509184394558e-01;
    Q->xi[5] = 5.74437169490815605442e-01;
    Q->xi[6] = 7.16697697064623595400e-01;
    Q->xi[7] = 8.39704784149512203117e-01;
    Q->xi[8] = 9.32531683344492255366e-01;
    Q->xi[9] = 9.86953264258585860039e-01;

    Q->w[0] = 3.33356721543440687968e-02;
    Q->w[1] = 7.47256745752902965729e-02;
    Q->w[2] = 1.09543181257991021998e-01;
    Q->w[3] = 1.34633359654998177546e-01;
    Q->w[4] = 1.47762112357376435087e-01;
    Q->w[5] = 1.47762112357376435087e-01;
    Q->w[6] = 1.34633359654998177546e-01;
    Q->w[7] = 1.09543181257991021998e-01;
    Q->w[8] = 7.47256745752902965729e-02;
    Q->w[9] = 3.33356721543440687968e-02;

    return;
}


void Regel11_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 21 */
quadrature *Q;
{
    Q->nop = 11;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 1.08856709269715035980e-02;
    Q->xi[1] = 5.64687001159523504624e-02;
    Q->xi[2] = 1.34923997212975337953e-01;
    Q->xi[3] = 2.40451935396594092037e-01;
    Q->xi[4] = 3.65228422023827513834e-01;
    Q->xi[5] = 5.00000000000000000000e-01;
    Q->xi[6] = 6.34771577976172486166e-01;
    Q->xi[7] = 7.59548064603405907963e-01;
    Q->xi[8] = 8.65076002787024662047e-01;
    Q->xi[9] = 9.43531299884047649538e-01;
    Q->xi[10] = 9.89114329073028496402e-01;

    Q->w[0] = 2.78342835580868332414e-02;
    Q->w[1] = 6.27901847324523123173e-02;
    Q->w[2] = 9.31451054638671257130e-02;
    Q->w[3] = 1.16596882295995239959e-01;
    Q->w[4] = 1.31402272255123331090e-01;
    Q->w[5] = 1.36462543388950315357e-01;
    Q->w[6] = 1.31402272255123331090e-01;
    Q->w[7] = 1.16596882295995239959e-01;
    Q->w[8] = 9.31451054638671257130e-02;
    Q->w[9] = 6.27901847324523123173e-02;
    Q->w[10] = 2.78342835580868332414e-02;

    return;
}


void Regel12_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 23 */
quadrature *Q;
{
    Q->nop = 12;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 9.21968287664037465473e-03;
    Q->xi[1] = 4.79413718147625716608e-02;
    Q->xi[2] = 1.15048662902847656482e-01;
    Q->xi[3] = 2.06341022856691276352e-01;
    Q->xi[4] = 3.16084250500909903124e-01;
    Q->xi[5] = 4.37383295744265542264e-01;
    Q->xi[6] = 5.62616704255734457736e-01;
    Q->xi[7] = 6.83915749499090096876e-01;
    Q->xi[8] = 7.93658977143308723648e-01;
    Q->xi[9] = 8.84951337097152343518e-01;
    Q->xi[10] = 9.52058628185237428339e-01;
    Q->xi[11] = 9.90780317123359625345e-01;

    Q->w[0] = 2.35876681932559135973e-02;
    Q->w[1] = 5.34696629976592154801e-02;
    Q->w[2] = 8.00391642716731131673e-02;
    Q->w[3] = 1.01583713361532960875e-01;
    Q->w[4] = 1.16746268269177404380e-01;
    Q->w[5] = 1.24573522906701392500e-01;
    Q->w[6] = 1.24573522906701392500e-01;
    Q->w[7] = 1.16746268269177404380e-01;
    Q->w[8] = 1.01583713361532960875e-01;
    Q->w[9] = 8.00391642716731131673e-02;
    Q->w[10] = 5.34696629976592154801e-02;
    Q->w[11] = 2.35876681932559135973e-02;

    return;
}


void Regel13_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 25 */
quadrature *Q;
{
    Q->nop = 13;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 7.90847264070592526359e-03;
    Q->xi[1] = 4.12008003885110173967e-02;
    Q->xi[2] = 9.92109546333450436029e-02;
    Q->xi[3] = 1.78825330279829889678e-01;
    Q->xi[4] = 2.75753624481776573561e-01;
    Q->xi[5] = 3.84770842022432602967e-01;
    Q->xi[6] = 5.00000000000000000000e-01;
    Q->xi[7] = 6.15229157977567397033e-01;
    Q->xi[8] = 7.24246375518223426439e-01;
    Q->xi[9] = 8.21174669720170110322e-01;
    Q->xi[10] = 9.00789045366654956397e-01;
    Q->xi[11] = 9.58799199611488982603e-01;
    Q->xi[12] = 9.92091527359294074736e-01;

    Q->w[0] = 2.02420023826579397600e-02;
    Q->w[1] = 4.60607499188642239572e-02;
    Q->w[2] = 6.94367551098936192318e-02;
    Q->w[3] = 8.90729903809728691400e-02;
    Q->w[4] = 1.03908023768444251156e-01;
    Q->w[5] = 1.13141590131448619206e-01;
    Q->w[6] = 1.16275776615436955097e-01;
    Q->w[7] = 1.13141590131448619206e-01;
    Q->w[8] = 1.03908023768444251156e-01;
    Q->w[9] = 8.90729903809728691400e-02;
    Q->w[10] = 6.94367551098936192318e-02;
    Q->w[11] = 4.60607499188642239572e-02;
    Q->w[12] = 2.02420023826579397600e-02;

    return;
}


void Regel14_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 27 */
quadrature *Q;
{
    Q->nop = 14;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->xi[0] = 6.85809565159383057920e-03;
    Q->xi[1] = 3.57825581682132413318e-02;
    Q->xi[2] = 8.63993424651175034051e-02;
    Q->xi[3] = 1.56353547594157264926e-01;
    Q->xi[4] = 2.42375681820922954017e-01;
    Q->xi[5] = 3.40443815536055119782e-01;
    Q->xi[6] = 4.45972525646328168967e-01;
    Q->xi[7] = 5.54027474353671831033e-01;
    Q->xi[8] = 6.59556184463944880218e-01;
    Q->xi[9] = 7.57624318179077045983e-01;
    Q->xi[10] = 8.43646452405842735074e-01;
    Q->xi[11] = 9.13600657534882496595e-01;
    Q->xi[12] = 9.64217441831786758668e-01;
    Q->xi[13] = 9.93141904348406169421e-01;

    Q->w[0] = 1.75597301658759315159e-02;
    Q->w[1] = 4.00790435798801049028e-02;
    Q->w[2] = 6.07592853439515923447e-02;
    Q->w[3] = 7.86015835790967672848e-02;
    Q->w[4] = 9.27691987389689068709e-02;
    Q->w[5] = 1.02599231860647801983e-01;
    Q->w[6] = 1.07631926731578895098e-01;
    Q->w[7] = 1.07631926731578895098e-01;
    Q->w[8] = 1.02599231860647801983e-01;
    Q->w[9] = 9.27691987389689068709e-02;
    Q->w[10] = 7.86015835790967672848e-02;
    Q->w[11] = 6.07592853439515923447e-02;
    Q->w[12] = 4.00790435798801049028e-02;
    Q->w[13] = 1.75597301658759315159e-02;

    return;
}


void Regel15_Legendre(Q)
/* exakt fuer Polynome bis zum Grad 29 */
quadrature *Q;
{
    Q->nop = 15;
    Q->xi = (double *) malloc(Q->nop * sizeof(double));
    Q->w = (double *) malloc(Q->nop * sizeof(double));

    Q->w[13] = 1.75597301658759315159e-02;

    Q->xi[0] = 6.00374098975728575522e-03;
    Q->xi[1] = 3.13633037996470478461e-02;
    Q->xi[2] = 7.58967082947863918997e-02;
    Q->xi[3] = 1.37791134319914976292e-01;
    Q->xi[4] = 2.14513913695730576231e-01;
    Q->xi[5] = 3.02924326461218315051e-01;
    Q->xi[6] = 3.99402953001282738850e-01;
    Q->xi[7] = 5.00000000000000000000e-01;
    Q->xi[8] = 6.00597046998717261150e-01;
    Q->xi[9] = 6.97075673538781684949e-01;
    Q->xi[10] = 7.85486086304269423769e-01;
    Q->xi[11] = 8.62208865680085023708e-01;
    Q->xi[12] = 9.24103291705213608100e-01;
    Q->xi[13] = 9.68636696200352952154e-01;
    Q->xi[14] = 9.93996259010242714245e-01;

    Q->w[0] = 1.53766209980586341773e-02;
    Q->w[1] = 3.51830237440540623546e-02;
    Q->w[2] = 5.35796102335859675059e-02;
    Q->w[3] = 6.97853389630771572239e-02;
    Q->w[4] = 8.31346029084969667766e-02;
    Q->w[5] = 9.30805000077811055134e-02;
    Q->w[6] = 9.92157426635557882281e-02;
    Q->w[7] = 1.01289120962780636440e-01;
    Q->w[8] = 9.92157426635557882281e-02;
    Q->w[9] = 9.30805000077811055134e-02;
    Q->w[10] = 8.31346029084969667766e-02;
    Q->w[11] = 6.97853389630771572239e-02;
    Q->w[12] = 5.35796102335859675059e-02;
    Q->w[13] = 3.51830237440540623546e-02;
    Q->w[14] = 1.53766209980586341773e-02;

    return;
}


void init_Gauss_Legendre(Q, g)
quadrature **Q;
unsigned int g;
{
/* Fehler-Routine */
    if (g > 15) {
        printf("g should be less equal 15\n");
        exit(0);
    }
/* Quadratur-Formeln bestimmen */
    (*Q) = (quadrature *) malloc(g * sizeof(quadrature));
    Regel01_Legendre(&(*Q)[0]);
    if (g == 1)
        return;
    Regel02_Legendre(&(*Q)[1]);
    if (g == 2)
        return;
    Regel03_Legendre(&(*Q)[2]);
    if (g == 3)
        return;
    Regel04_Legendre(&(*Q)[3]);
    if (g == 4)
        return;
    Regel05_Legendre(&(*Q)[4]);
    if (g == 5)
        return;
    Regel06_Legendre(&(*Q)[5]);
    if (g == 6)
        return;
    Regel07_Legendre(&(*Q)[6]);
    if (g == 7)
        return;
    Regel08_Legendre(&(*Q)[7]);
    if (g == 8)
        return;
    Regel09_Legendre(&(*Q)[8]);
    if (g == 9)
        return;
    Regel10_Legendre(&(*Q)[9]);
    if (g == 10)
        return;
    Regel11_Legendre(&(*Q)[10]);
    if (g == 11)
        return;
    Regel12_Legendre(&(*Q)[11]);
    if (g == 12)
        return;
    Regel13_Legendre(&(*Q)[12]);
    if (g == 13)
        return;
    Regel14_Legendre(&(*Q)[13]);
    if (g == 14)
        return;
    Regel15_Legendre(&(*Q)[14]);
    if (g == 15)
        return;
}


void free_Gauss_Legendre(Q, g)
quadrature **Q;
unsigned int g;
{
    unsigned int k;

    for (k = 0; k < g; k++) {
        free((*Q)[k].xi);
        free((*Q)[k].w);
    }
    free(*Q);

    return;
}
/* warning-disabler-start */

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#pragma warning pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

/* warning-disabler-end */

