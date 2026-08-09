import { http, HttpResponse } from 'msw';

/**
 * Example handler for the GitHub profile linked from the Sidebar
 * (https://github.com/acfharbinger). Not currently fetched by the app —
 * this documents the pattern integration tests should follow if a future
 * feature (e.g. live follower/stats data) adds a real network call, and
 * gives the MSW server at least one realistic handler to exercise.
 */
export const handlers = [
  http.get('https://api.github.com/users/acfharbinger', () => {
    return HttpResponse.json({
      login: 'acfharbinger',
      name: 'ACFHarbinger',
      public_repos: 42,
      followers: 7,
    });
  }),
];
