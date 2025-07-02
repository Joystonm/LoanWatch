import React, { useState } from 'react';

const NotificationBell = () => {
  const [showNotifications, setShowNotifications] = useState(false);
  
  // Mock notifications data
  const notifications = [
    {
      id: 1,
      type: 'alert',
      message: 'Potential bias detected in recent predictions',
      time: '10 minutes ago',
      read: false
    },
    {
      id: 2,
      type: 'info',
      message: 'Model retraining completed successfully',
      time: '2 hours ago',
      read: false
    },
    {
      id: 3,
      type: 'success',
      message: 'Fairness metrics improved by 15%',
      time: '1 day ago',
      read: true
    }
  ];
  
  const unreadCount = notifications.filter(n => !n.read).length;
  
  const getNotificationIcon = (type) => {
    switch (type) {
      case 'alert':
        return (
          <div className="notification-icon-wrapper alert">
            <svg className="notification-icon" width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
        );
      case 'info':
        return (
          <div className="notification-icon-wrapper info">
            <svg className="notification-icon" width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
        );
      case 'success':
        return (
          <div className="notification-icon-wrapper success">
            <svg className="notification-icon" width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
        );
      default:
        return null;
    }
  };
  
  return (
    <div className="notification-bell">
      <button
        onClick={() => setShowNotifications(!showNotifications)}
        className="notification-button"
        aria-label="View notifications"
      >
        <svg className="notification-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
        </svg>
        
        {unreadCount > 0 && (
          <span className="notification-badge">
            {unreadCount}
          </span>
        )}
      </button>
      
      {showNotifications && (
        <div className="notification-dropdown">
          <div className="notification-header">
            <h3 className="notification-header-title">Notifications</h3>
            <button className="notification-header-action">
              Mark all as read
            </button>
          </div>
          
          <div className="notification-body">
            {notifications.length === 0 ? (
              <div className="notification-empty">
                No notifications
              </div>
            ) : (
              <ul className="notification-list">
                {notifications.map((notification) => (
                  <li 
                    key={notification.id} 
                    className={`notification-item ${notification.read ? '' : 'unread'}`}
                  >
                    {getNotificationIcon(notification.type)}
                    <div className="notification-content">
                      <p className="notification-message">{notification.message}</p>
                      <p className="notification-time">{notification.time}</p>
                    </div>
                    <div className="notification-actions">
                      <button className="notification-close">
                        <span className="sr-only">Close</span>
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          
          <div className="notification-footer">
            <button className="notification-footer-action">
              View all notifications
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotificationBell;
