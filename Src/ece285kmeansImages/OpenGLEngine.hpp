// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

const int trainSize = 60000;
const int testSize = 10000;
const int n_rows = 28;
const int n_cols = 28;
const int dim = n_rows*n_cols;
const int k = 10; // Number of Means to be used for clustering
const int number_of_iterations = 5;

float zoom = 15;
float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

#define MAX(a,b) ((a > b) ? a : b)

const size_t lengthOfWait = 20; //in frames

size_t currentState = 0; // which part of looping animation I am watching
size_t frameInState = 0; //number of frames spent in the state

//static std::vector <std::tuple<int, float, float, float>> kmeansContainer;
static std::vector <int> assignmentContainer; //
static std::vector <std::tuple<float, float, float>> dataContainer;

void fillTest(size_t sizeOfClass, size_t num_iterations) {
	dataContainer.clear();
	assignmentContainer.clear();
	srand(time(NULL));
	for (int i = 0; i < sizeOfClass; i++) {
		dataContainer.push_back(std::make_tuple(((float)rand()) / RAND_MAX, ((float)rand()) / RAND_MAX, ((float)rand()) / RAND_MAX));
	}
	for (int i = 0; i < sizeOfClass*num_iterations; i++) {
		assignmentContainer.push_back(rand());
	}
}

void checkAssignment() {
	int c1 = 0, c2 = 0, c3 = 0;
	for (int i = 0; i < assignmentContainer.size(); i++) {
		int temp = assignmentContainer[i];
		if (temp == 0) {
			c1++;
		}
		else if (temp == 1) {
			c2++;
		}
		else c3++;
	}
	printf("%d,%d,%d\n", c1, c2, c3);
}

void colorMap(int key) {
    //printf("label: %d\n", key);
	if (key==0) glColor3f(1.0f, 1.0f, 1.0f);
	else if (key == 1) glColor3f(0.0f, 1.0f, 0.0f);
	else if (key == 2) glColor3f(0.0f, 0.0f, 1.0f);
	else if (key == 3) glColor3f(1.0f, 0.0f, 0.0f);
	else if (key == 4) glColor3f(1.0f, 1.0f, 0.0f);
	else if (key == 5) glColor3f(0.0f, 1.0f, 1.0f);
	else if (key == 6) glColor3f(1.0f, 0.0f, 1.0f);
	else if (key == 7) glColor3f(0.5f, 0.7f, 1.0f);
	else if (key == 8) glColor3f(1.0f, 0.7f, 0.5f);
	else if (key == 9) glColor3f(1.0f, 1.0f, 1.0f);
}


void MouseWheel(int wheel, int direction, int x, int y)
{
	wheel = 0;
	if (direction == -1)
	{
		zoom -= 0.5;

	}
	else if (direction == +1)
	{
		zoom += 0.5;
	}

	glutPostRedisplay();

}


void render() {
	if (frameInState > lengthOfWait) {
		currentState++; frameInState = 0; //printf("Current state:%d\n",currentState);
	}
	if (currentState >= number_of_iterations) currentState = 0;

	glScalef(zoom, zoom, zoom);

	glVertexPointer(4, GL_FLOAT, 0, 0);
	// Draw a triangle:

	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(-20.0f, 0.0f, 0.0f);
	glVertex3f(20.0f, 0.0f, 0.0f);
	glVertex3f(0.0f,-20.0f, 0.0f);
	glVertex3f(0.0f,20.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, -20.0f);
	glVertex3f(0.0f, 0.0f, 20.0f);
	glEnd();
	glFlush();

	glBegin(GL_POINTS);

	//printf("%d\n", assignmentContainer.size() / dataContainer.size());

	for (int i = 0; i < dataContainer.size(); i++) {
		//printf("%d\n", std::get<0>(dataContainer[i]));
		colorMap(assignmentContainer[i + currentState*trainSize]);
		//printf("x:%f,y:%f,z:%f\n", std::get<0>(dataContainer[i]), std::get<1>(dataContainer[i]), std::get<2>(dataContainer[i]));
		glVertex3f(std::get<0>(dataContainer[i]), std::get<1>(dataContainer[i]), std::get<2>(dataContainer[i]));
	}
	// Lower left vertex
	//glVertex3f(-1.0f, -0.5f, -3.0f);

	// Lower right vertex
	//glVertex3f(1.0f, -0.5f, -4.0f);

	// Upper vertex
	//glVertex3f(0.0f, 0.5f, -2.0f);
	glEnd();
	frameInState++;
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "K-means Visualization: %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}

void display()
{
	//printf("zoom: %d\n", zoom);
	sdkStartTimer(&timer);
	//gluLookAt(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0);
	// run CUDA kernel to generate vertex positions
	//runCuda(&cuda_vbo_resource);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);


	render();
	// render from the vbo
	//glBindBuffer(GL_ARRAY_BUFFER, vbo);

	//glEnableClientState(GL_VERTEX_ARRAY);
	//
	//glColor3f(1.0, 0.0, 0.0);
	//glVertex3f(0.5, 0.5, 0.5);
	//glDrawArrays(GL_POINTS, 0, 1);
	//glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	g_fAnim += 0.01f;

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}


void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27):
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	}
}

class GraphicsController {
protected:
	unsigned int window_width;
	unsigned int window_height;



public:
	GraphicsController(unsigned int aWidth = 512, unsigned int aHeight = 512) : window_width(aWidth), window_height(aHeight) {
#if defined(__linux__)
		setenv("DISPLAY", ":0", 0);
#endif
	}

	bool initGL(int* argc, char **argv) {
		//fillTest(trainSize,number_of_iterations);
		//checkAssignment();
		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize(window_width, window_height);
		glutCreateWindow("ECE 285 K-means Visualization");
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMotionFunc(motion);
		glutMouseFunc(mouse);
		glutMouseWheelFunc(MouseWheel);
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

		// initialize necessary OpenGL extensions
		if (!isGLVersionSupported(2, 0))
		{
			fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
			fflush(stderr);
			return false;
		}

		// default initialization
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glLineWidth(3);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_POINT_SMOOTH);

		// viewport
		glViewport(0, 0, window_width, window_height);

		// projection
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.001, 200.0);

		SDK_CHECK_ERROR_GL();

		return true;
	}



	void run() {
		glutMainLoop();
	}
};
