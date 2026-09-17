import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    title: 'mdebug',
    loadComponent: () =>
      import('./features/landing/landing.component').then((m) => m.LandingComponent),
  },
  {
    path: 'debug/:sessionId',
    title: 'mdebug — session',
    loadComponent: () =>
      import('./features/debugger/debugger-page.component').then(
        (m) => m.DebuggerPageComponent,
      ),
  },
  { path: '**', redirectTo: '' },
];
