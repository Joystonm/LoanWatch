function Dashboard() {
  // Mock data for dashboard
  const metrics = [
    { title: 'Total Applications', value: '10,000', change: '+12%', changeType: 'positive' },
    { title: 'Approval Rate', value: '64.0%', change: '+2.5%', changeType: 'positive' },
    { title: 'Average Score', value: '72.3', change: '+1.2', changeType: 'positive' },
    { title: 'Fairness Index', value: '0.86', change: '+0.04', changeType: 'positive' },
  ];
  
  const recentApplications = [
    { id: 'APP-1234', name: 'John Smith', amount: '$245,000', status: 'Approved', date: '2023-06-15' },
    { id: 'APP-1235', name: 'Sarah Johnson', amount: '$180,000', status: 'Denied', date: '2023-06-14' },
    { id: 'APP-1236', name: 'Michael Brown', amount: '$320,000', status: 'Pending', date: '2023-06-14' },
    { id: 'APP-1237', name: 'Emily Davis', amount: '$210,000', status: 'Approved', date: '2023-06-13' },
    { id: 'APP-1238', name: 'Robert Wilson', amount: '$175,000', status: 'Denied', date: '2023-06-12' },
  ];
  
  return (
    <div className="space-y-6">
      {/* Welcome Card */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-700 rounded-xl shadow-md overflow-hidden">
        <div className="px-6 py-8 md:px-8 md:py-10">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div className="mb-6 md:mb-0">
              <h2 className="text-2xl font-bold text-white mb-2">Welcome to LoanWatch</h2>
              <p className="text-primary-100 max-w-2xl">
                Monitor loan approval predictions, analyze fairness metrics, and ensure regulatory compliance with our AI-powered platform.
              </p>
            </div>
            <div className="flex space-x-3">
              <button className="btn bg-white text-primary-700 hover:bg-primary-50">
                New Application
              </button>
              <button className="btn bg-primary-500 text-white hover:bg-primary-400 border border-primary-400">
                View Reports
              </button>
            </div>
          </div>
        </div>
      </div>
      
      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => (
          <div key={index} className="bg-white rounded-xl shadow-card overflow-hidden animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
            <div className="px-6 py-5">
              <p className="text-sm font-medium text-neutral-500">{metric.title}</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-2xl font-semibold text-neutral-900">{metric.value}</p>
                <p className={`ml-2 flex items-baseline text-sm font-semibold ${
                  metric.changeType === 'positive' ? 'text-success-600' : 'text-danger-600'
                }`}>
                  {metric.changeType === 'positive' ? (
                    <svg className="self-center flex-shrink-0 h-4 w-4 text-success-500" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                      <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    <svg className="self-center flex-shrink-0 h-4 w-4 text-danger-500" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                      <path fillRule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                  <span className="sr-only">{metric.changeType === 'positive' ? 'Increased' : 'Decreased'} by</span>
                  {metric.change}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Recent Applications */}
      <div className="bg-white rounded-xl shadow-card overflow-hidden">
        <div className="px-6 py-5 border-b border-neutral-200">
          <h3 className="text-lg font-medium text-neutral-900">Recent Applications</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-neutral-200">
            <thead className="bg-neutral-50">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                  Application ID
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                  Applicant
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                  Amount
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                  Status
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                  Date
                </th>
                <th scope="col" className="relative px-6 py-3">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-neutral-200">
              {recentApplications.map((application, index) => (
                <tr key={index} className="hover:bg-neutral-50 transition-colors duration-150 ease-in-out">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-neutral-900">
                    {application.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-700">
                    {application.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-700">
                    {application.amount}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      application.status === 'Approved' 
                        ? 'bg-success-100 text-success-800' 
                        : application.status === 'Denied'
                          ? 'bg-danger-100 text-danger-800'
                          : 'bg-warning-100 text-warning-800'
                    }`}>
                      {application.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-700">
                    {application.date}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <a href="#" className="text-primary-600 hover:text-primary-900">View</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="px-6 py-4 border-t border-neutral-200 bg-neutral-50">
          <a href="#" className="text-sm font-medium text-primary-600 hover:text-primary-900">
            View all applications
            <span aria-hidden="true"> &rarr;</span>
          </a>
        </div>
      </div>
      
      {/* Fairness Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-card overflow-hidden">
          <div className="px-6 py-5 border-b border-neutral-200">
            <h3 className="text-lg font-medium text-neutral-900">Fairness Summary</h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-neutral-700">Gender Fairness</span>
                  <span className="text-neutral-900">86%</span>
                </div>
                <div className="w-full bg-neutral-100 rounded-full h-2">
                  <div className="bg-primary-600 h-2 rounded-full" style={{ width: '86%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-neutral-700">Race Fairness</span>
                  <span className="text-neutral-900">78%</span>
                </div>
                <div className="w-full bg-neutral-100 rounded-full h-2">
                  <div className="bg-warning-500 h-2 rounded-full" style={{ width: '78%' }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-neutral-700">Age Fairness</span>
                  <span className="text-neutral-900">92%</span>
                </div>
                <div className="w-full bg-neutral-100 rounded-full h-2">
                  <div className="bg-success-500 h-2 rounded-full" style={{ width: '92%' }}></div>
                </div>
              </div>
            </div>
            <div className="mt-6">
              <a href="#" className="text-sm font-medium text-primary-600 hover:text-primary-900" onClick={(e) => {
                e.preventDefault();
                // Handle navigation to fairness page
              }}>
                View detailed fairness report
                <span aria-hidden="true"> &rarr;</span>
              </a>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-xl shadow-card overflow-hidden">
          <div className="px-6 py-5 border-b border-neutral-200">
            <h3 className="text-lg font-medium text-neutral-900">Regulatory Compliance</h3>
          </div>
          <div className="p-6">
            <div className="flex items-center mb-6">
              <div className="h-16 w-16 rounded-full bg-success-100 flex items-center justify-center">
                <svg className="h-8 w-8 text-success-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <div className="ml-4">
                <h4 className="text-lg font-medium text-neutral-900">Compliant</h4>
                <p className="text-sm text-neutral-500">All regulatory requirements are being met</p>
              </div>
            </div>
            <ul className="space-y-3">
              <li className="flex items-start">
                <svg className="h-5 w-5 text-success-500 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="ml-2 text-sm text-neutral-700">Equal Credit Opportunity Act (ECOA)</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-success-500 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="ml-2 text-sm text-neutral-700">Fair Housing Act (FHA)</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-success-500 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="ml-2 text-sm text-neutral-700">Community Reinvestment Act (CRA)</span>
              </li>
            </ul>
            <div className="mt-6">
              <a href="#" className="text-sm font-medium text-primary-600 hover:text-primary-900">
                View compliance details
                <span aria-hidden="true"> &rarr;</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
