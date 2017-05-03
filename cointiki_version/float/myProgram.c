#include "contiki.h"
#include "floattostring.c"
PROCESS(floating_operation, "testing operations");
AUTOSTART_PROCESSES(&floating_operation);

static float a=2.3;
static float b=1.2;
static float d=2.3;

PROCESS_THREAD(floating_operation, ev, data)
{
  PROCESS_BEGIN();
  	float c;
  	char res[20];
  	//float n = 233.007;
  	while(1){


  			c = a + b;
  		    ftoa(c, res, 2);
  		    printf("2.3 + 1.2  = \n\"%s\"\n", res);
  		    c= d/a;
  		    ftoa(c, res, 4);
  		    printf("2.3 / 2.3 = \n\"%s\"\n", res);
  		    c= a*b;
  		    ftoa(c, res, 3);
  		    printf("2.3*1.2 = \n\"%s\"\n", res);

  	}

  	PROCESS_END();
}
