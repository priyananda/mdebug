import {
  ApplicationConfig,
  provideExperimentalZonelessChangeDetection,
} from '@angular/core';
import { provideHttpClient, withFetch } from '@angular/common/http';
import {
  provideRouter,
  withComponentInputBinding,
  withHashLocation,
  withInMemoryScrolling,
} from '@angular/router';

import { environment } from '../environments/environment';
import { HttpInferenceApi } from './core/api/http/http-inference-api';
import { InferenceApi } from './core/api/inference-api';
import { MockInferenceApi } from './core/api/mock/mock-inference-api';
import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    // All state in this app is signals, and the engine ticks on timers at 30-60Hz.
    // Under Zone.js every one of those timers would trigger an app-wide check.
    provideExperimentalZonelessChangeDetection(),
    provideHttpClient(withFetch()),
    // The seam. Everything above `InferenceApi` is identical either way: the
    // UI reduces the same event stream whichever implementation produces it.
    // `useExisting` means only the selected one is ever instantiated.
    MockInferenceApi,
    HttpInferenceApi,
    {
      provide: InferenceApi,
      useExisting: environment.useMock ? MockInferenceApi : HttpInferenceApi,
    },
    // Hash routing: GitHub Pages has no SPA rewrite, so a path-routed deep link
    // (/debug/:id) 404s on refresh.
    provideRouter(
      routes,
      withHashLocation(),
      withComponentInputBinding(),
      withInMemoryScrolling({ scrollPositionRestoration: 'top' }),
    ),
  ],
};
