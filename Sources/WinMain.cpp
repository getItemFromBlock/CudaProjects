#include <iostream>
#include <windowsx.h>
#include <Windows.h>
#include "Types.hpp"

#include "RenderThread.hpp"

#ifdef _DEBUG
#include <crtdbg.h>
#endif

CHAR szClassName[] = "MainClass";
CHAR szTitle[] = "CUDA Demo";
HCURSOR cursorHide, cursorDefault;
bool captured = false;
RenderThread th;

LRESULT CALLBACK WndProc(_In_ HWND hWnd, _In_ UINT message, _In_ WPARAM wParam, _In_ LPARAM lParam);
void SetKeyState(WPARAM wParam, bool pressed);
void OnMoveMouse(HWND hwnd, bool reset = false);

int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR pCmdLine, _In_ int nCmdShow)
{
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    //_CrtSetBreakAlloc(163);
#endif
    {
        cursorDefault = LoadCursor(NULL, IDC_ARROW);
        cursorHide = nullptr;

        WNDCLASSEX wcex = {};
        wcex.cbSize = sizeof(WNDCLASSEX);
        wcex.style = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc = WndProc;
        wcex.cbClsExtra = 0;
        wcex.cbWndExtra = 0;
        wcex.hInstance = hInstance;
        wcex.hIcon = LoadIcon(wcex.hInstance, IDI_APPLICATION);
        wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
        wcex.hbrBackground = NULL;
        wcex.lpszMenuName = NULL;
        wcex.lpszClassName = szClassName;
        wcex.hIconSm = LoadIcon(wcex.hInstance, IDI_APPLICATION);

        if (!RegisterClassEx(&wcex))
        {
            MessageBox(NULL, "Call to RegisterClassExW failed!", szTitle, NULL);
            return 1;
        }

        if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_SYSTEM_AWARE))
        {
            MessageBox(NULL, "Could not set window dpi awareness !", szTitle, NULL);
        }
        HWND hWnd = CreateWindowEx(WS_EX_OVERLAPPEDWINDOW, szClassName, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 450, NULL, NULL, hInstance, NULL);

        ShowWindow(hWnd, nCmdShow);
        UpdateWindow(hWnd);

        th.Init(hWnd, Maths::IVec2(800, 600), true);
        LONG_PTR lExStyle = GetWindowLongPtr(hWnd, GWL_EXSTYLE);
        lExStyle &= ~(WS_EX_DLGMODALFRAME | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE);
        SetWindowLongPtr(hWnd, GWL_EXSTYLE, lExStyle);
        SetWindowPos(hWnd, NULL, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER);
        if (!hWnd)
        {
            MessageBox(NULL, "Call to CreateWindow failed!", szTitle, NULL);
            return 1;
        }

        // Main message loop:
        MSG msg;
        while (GetMessageW(&msg, NULL, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        th.Quit();
        return (int)msg.wParam;
    }
}

LRESULT CALLBACK WndProc(_In_ HWND hWnd, _In_ UINT message, _In_ WPARAM wParam, _In_ LPARAM lParam)
{
    switch (message)
    {
    case WM_PAINT:
        break;
    case WM_CLEAR:
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_SIZE:
        th.Resize(Maths::IVec2(LOWORD(lParam), HIWORD(lParam)));
        break;
    case WM_KEYDOWN:
        SetKeyState(wParam, true);
        break;
    case WM_KEYUP:
        if (wParam == VK_ESCAPE)
        {
            captured = !captured;
            SetCursor(captured ? cursorHide : cursorDefault);
            if (captured)
            {
                OnMoveMouse(hWnd, true);
            }
            break;
        }
        SetKeyState(wParam, false);
        break;
    case WM_SYSKEYDOWN:
        return DefWindowProc(hWnd, message, wParam, lParam);
    case WM_MOUSEMOVE:
        if (captured) OnMoveMouse(hWnd);
        break;
    case WM_SETCURSOR:
    {
        SetCursor(captured ? cursorHide : cursorDefault);
        break;
    }
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

void SetKeyState(WPARAM wParam, bool pressed)
{
    switch (wParam)
    {
    case 'A':
        th.SetKeyState(0, pressed);
        break;
    case 'Q':
        th.SetKeyState(1, pressed);
        break;
    case 'W':
        th.SetKeyState(2, pressed);
        break;
    case 'D':
        th.SetKeyState(3, pressed);
        break;
    case 'E':
        th.SetKeyState(4, pressed);
        break;
    case 'S':
        th.SetKeyState(5, pressed);
        break;
    case VK_UP:
        th.SetKeyState(6, pressed);
        break;
    case VK_DOWN:
        th.SetKeyState(7, pressed);
        break;
    case VK_RETURN:
        th.SetKeyState(8, pressed);
        break;
    default:
        break;
    }
}

void OnMoveMouse(HWND hwnd, bool reset)
{
    RECT rect = {};
    POINT mPos = {};
    GetCursorPos(&mPos);
    GetWindowRect(hwnd, &rect);
    int tempX = rect.left + (rect.right - rect.left) / 2;
    int tempY = rect.top + (rect.bottom - rect.top) / 2;
    if (mPos.x != tempX || mPos.y != tempY) {
        int rPosX = mPos.x - tempX;
        int rPosY = mPos.y - tempY;
        if (!reset) th.MoveMouse(Maths::Vec2(static_cast<f32>(rPosX), static_cast<f32>(rPosY)));
        SetCursorPos(tempX, tempY);
    }
}