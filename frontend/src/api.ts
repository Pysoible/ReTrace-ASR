export const api = (path: string) => path.startsWith('/api/') ? path : `/api/${path}`;
