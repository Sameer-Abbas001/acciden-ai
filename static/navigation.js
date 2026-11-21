'use strict';
// Change the visible dashboard without opening a tab or reloading the camera.
window.AccidenNavigation = {
  navigate(url) {
    const destination = new URL(url, location.origin);
    if (destination.origin !== location.origin) throw new Error('Dashboard must use this deployment');
    const route = destination.pathname.replace(/\.html$/, '').replace(/\/$/, '');
    const view = route === '/ambulance' ? 'ambulance' : route === '/police' ? 'police' : 'vision';
    const path = view === 'vision' ? '/admin' : '/' + view;
    const incident = destination.searchParams.get('incident');
    const target = path + (incident ? '?incident=' + encodeURIComponent(incident) : '');
    if (location.pathname + location.search !== target) history.pushState({}, '', target);
    window.dispatchEvent(new CustomEvent('acciden:navigate', {detail:{view, incident}}));
  }
};
