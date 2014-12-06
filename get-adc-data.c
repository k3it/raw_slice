#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>

#include <byteswap.h>

int create_raw_socket()
{
        int fd;
        fd = socket(PF_PACKET, SOCK_RAW, htons(0xEFFF));
        return fd;
}



int
main(int argc, char **argv)
{
  int fd = create_raw_socket();
  if (fd < 0)
    {
        fprintf(stderr, "Error creating raw socket\n");
    }
    else
    {
  	fprintf(stderr, "created raw socket\n");
    }

    char  data[10000];
    short samples[4500];

    int skip = 1456;
    unsigned short lastno=0;
    unsigned int packetno=0;


    ssize_t             n;

    for (;;)
    {
     n = recv(fd, &data, sizeof data, 0);
     if (n == -1) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "Receive error: %s.\n", strerror(errno));
            break;
     }

     //fprintf( stderr, "received %d bytes\n", n);

     // int k = 0;
     // for (int ax = 14; ax < n; ax += 2) 
     // {
     //  halfdata[k++] = data[ax];
     // }

     fwrite(data+14,sizeof(unsigned char), n-14, stdout);

     //memcpy(samples, data+14, 9000);
     //packetno++;

     // if (packetno == 16000) 
     // {
     //    for (int i=0; i < 4500; i++) {
     //      printf("%d ", __bswap_16(samples[i]));
     //    }
     //    printf("\npacket no: %d \n\n", packetno);
     //    packetno = 0;
     //  }
      
     //fwrite(halfdata,sizeof(unsigned char), (n-14)/2, stdout);
     //fflush(stdout);
    }
}
  
    
    

