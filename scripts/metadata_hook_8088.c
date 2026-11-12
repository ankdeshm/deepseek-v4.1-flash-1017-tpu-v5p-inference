#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

static int (*real_connect)(int, const struct sockaddr *, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    if (addr && addr->sa_family == AF_INET && addrlen >= sizeof(struct sockaddr_in)) {
        const struct sockaddr_in *orig = (const struct sockaddr_in *)addr;
        char ip[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &(orig->sin_addr), ip, INET_ADDRSTRLEN) != NULL) {
            if (strcmp(ip, "169.254.169.254") == 0 && ntohs(orig->sin_port) == 80) {
                const char *port_env = getenv("MOCK_METADATA_PORT");
                int target_port = (port_env && *port_env) ? atoi(port_env) : 8088;
                struct sockaddr_in redirected;
                memcpy(&redirected, orig, sizeof(struct sockaddr_in));
                redirected.sin_addr.s_addr = inet_addr("127.0.0.1");
                redirected.sin_port = htons((uint16_t)target_port);
                const char *verbose = getenv("MOCK_METADATA_VERBOSE");
                if (verbose && strcmp(verbose, "1") == 0) {
                    char msg[160];
                    int n = snprintf(
                        msg, sizeof(msg),
                        "[metadata_hook_8088 pid=%d] 169.254.169.254:80 -> 127.0.0.1:%d\n",
                        (int)getpid(), target_port
                    );
                    if (n > 0) {
                        (void)write(2, msg, (size_t)n);
                    }
                }
                return real_connect(sockfd, (const struct sockaddr *)&redirected, sizeof(redirected));
            }
        }
    }
    return real_connect(sockfd, addr, addrlen);
}
