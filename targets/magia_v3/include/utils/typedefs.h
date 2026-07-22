#ifndef TYPEDEFS_H_
#define TYPEDEFS_H_

// GCC_MULTILIB doesn't know what a float16 is.
#if CV32E40P
typedef _Float16 float16;
#endif

#endif /* TYPEDEFS_H_ */
